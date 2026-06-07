"""
GRPO cold-start training on the Countdown arithmetic puzzle (TinyZero-style).

Uses Qwen2.5-3B BASE model (not instruct) so reasoning emerges from scratch.
The model starts with no chain-of-thought and develops <think>/<answer> structure
through the format + accuracy reward signal — the before/after is the demo.

Data contract (same as MuJoCo scripts):
  - heartbeat.json written every 60s (via HeartbeatWriter)
  - honours Sentinel nudges (results/{agent_id}/nudge.json)
  - writes eval_result.json on completion
  - writes baseline_responses.json (pre-training) and inference_results.json (post-training)

Run from /workspace (repo root on the pod), e.g.:
    /workspace/venv/bin/python training/train_grpo_countdown.py \
        --agent-id agent_4 --time-budget 1200 --lr 1e-6 --seed 42 \
        --results-dir /workspace/results
"""

import argparse
import json
import os
import sys
import time

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PKG_ROOT, ".env"))
except ImportError:
    pass

import torch
import weave
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from environments.countdown_env import (
    SYSTEM_PROMPT,
    accuracy_reward_fn,
    evaluate_solution,
    format_reward_fn,
    generate_countdown_prompt,
    load_countdown_dataset,
)
from training.callbacks.heartbeat_writer import HeartbeatWriter

# Base model — no RLHF constraints, reasoning emerges from scratch via GRPO
MODEL_NAME = "Qwen/Qwen2.5-3B"


def init_weave(agent_id: str):
    if os.environ.get("WEAVE_DISABLED"):
        return
    if not os.environ.get("WANDB_API_KEY"):
        print("[weave] WANDB_API_KEY not set — tracing skipped")
        return
    project = os.environ.get("WEAVE_PROJECT", "autorl")
    try:
        weave.init(project)
        print(f"[weave] tracing to project '{project}'")
    except Exception as e:
        print(f"[weave] init skipped ({e})")


class HeartbeatTrainerCallback(TrainerCallback):
    def __init__(self, hb: HeartbeatWriter):
        self.hb = hb

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.log_history:
            return
        recent = state.log_history[-5:]
        mean_reward = sum(log.get("reward", 0.0) for log in recent) / max(len(recent), 1)
        last_loss = recent[-1].get("loss")
        self.hb.update(state.global_step, mean_reward, loss=last_loss)


class TimeBudgetCallback(TrainerCallback):
    def __init__(self, start_time: float, time_budget: float):
        self.start_time = start_time
        self.time_budget = time_budget

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() - self.start_time >= self.time_budget:
            control.should_training_stop = True
        return control


class NudgeCallback(TrainerCallback):
    def __init__(self, hb: HeartbeatWriter, agent_id: str):
        self.hb = hb
        self.agent_id = agent_id
        self._trainer = None

    def set_trainer(self, trainer):
        self._trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        nudge = self.hb.check_nudge()
        if nudge and self._trainer is not None:
            new_lr = nudge.get("lr", args.learning_rate)
            for pg in self._trainer.optimizer.param_groups:
                pg["lr"] = new_lr
            print(f"[{self.agent_id}] Nudged: lr={new_lr}")
        return control


def _run_inference(model, tokenizer, cases: list, agent_id: str, label: str) -> list[dict]:
    """Generate responses for a list of test cases and return structured results."""
    results = []
    model.eval()
    for row in cases:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": generate_countdown_prompt(row["nums"], row["target"])},
        ]
        inputs = tokenizer.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True, return_dict=True
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        score = evaluate_solution(response, row["target"], row["nums"])
        results.append({
            "numbers": row["nums"],
            "target": row["target"],
            "model_response": response,
            "success": score == 1.0,
        })
        tag = "ok" if score == 1.0 else "fail"
        print(f"  [{label}] [{row['nums']} -> {row['target']}] {tag}")
    return results


def train_grpo(agent_id, time_budget, lr, seed, num_generations, temperature, results_dir, device="auto"):
    """Time-budgeted GRPO training on Countdown."""
    os.makedirs(f"{results_dir}/{agent_id}", exist_ok=True)

    hb = HeartbeatWriter(agent_id, results_dir)
    hb.start()

    print(f"[{agent_id}] Loading model {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16,
        ).to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype="auto" if device == "auto" else None,
        )
        if device == "auto" and torch.cuda.is_available():
            model = model.to("cuda")
        elif device not in ("auto", "cpu"):
            model = model.to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    # --- Pre-training baseline: record 5 responses before any training ---
    print(f"[{agent_id}] Recording pre-training baseline (5 examples)...")
    test_dataset = load_countdown_dataset(split="test", seed=seed)
    showcase_cases = list(test_dataset)[:5]
    baseline_results = _run_inference(model, tokenizer, showcase_cases, agent_id, "pre")
    baseline_path = f"{results_dir}/{agent_id}/baseline_responses.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline_results, f, indent=2)
    print(f"[{agent_id}] Baseline saved: {baseline_path}")

    print(f"[{agent_id}] Loading dataset...")
    dataset = load_countdown_dataset(split="train", seed=seed)
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(min(1_500, len(dataset))))  # keep init fast; 1500 rows >> steps in budget

    def format_row(row):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": generate_countdown_prompt(row["nums"], row["target"])},
            ],
            "target": row["target"],
            "numbers": row["nums"],
        }

    formatted = dataset.map(format_row, remove_columns=dataset.column_names, num_proc=4)

    _cuda_ok = torch.cuda.is_available()
    _bf16_ok = _cuda_ok and torch.cuda.is_bf16_supported()
    _default_bs = 2 if device == "mps" else 4
    _batch_size = max(_default_bs, num_generations)
    grpo_config = GRPOConfig(
        use_vllm=True,
        learning_rate=lr,
        per_device_train_batch_size=_batch_size,
        num_generations=num_generations,
        max_completion_length=256,
        temperature=temperature,
        seed=seed,
        output_dir=f"{results_dir}/{agent_id}",
        logging_steps=1,
        save_steps=9999,
        max_steps=10_000,  # safety ceiling; TimeBudgetCallback stops earlier
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        bf16=_bf16_ok,
        fp16=_cuda_ok and not _bf16_ok,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=formatted,
        reward_funcs=[format_reward_fn, accuracy_reward_fn],
        peft_config=lora_config,
    )

    start = time.time()

    print(f"[{agent_id}] Starting GRPO training (budget: {time_budget}s, lr={lr})...")
    nudge_cb = NudgeCallback(hb, agent_id)
    trainer.add_callback(HeartbeatTrainerCallback(hb))
    trainer.add_callback(TimeBudgetCallback(start, time_budget))
    trainer.add_callback(nudge_cb)
    nudge_cb.set_trainer(trainer)
    trainer.train()

    recent = trainer.state.log_history[-5:] if trainer.state.log_history else []
    mean_reward = sum(log.get("reward", 0.0) for log in recent) / max(len(recent), 1)
    step = trainer.state.global_step

    # --- Post-training evaluation on same 5 cases (for before/after comparison) ---
    print(f"[{agent_id}] Running post-training inference on same 5 cases...")
    inference_results = _run_inference(model, tokenizer, showcase_cases, agent_id, "post")
    infer_path = f"{results_dir}/{agent_id}/inference_results.json"
    with open(infer_path, "w") as f:
        json.dump(inference_results, f, indent=2)
    print(f"[{agent_id}] Inference results saved: {infer_path}")

    # --- Accuracy eval on held-out test set ---
    print(f"[{agent_id}] Evaluating on test set...")
    total = min(10, len(test_dataset))
    correct = sum(
        1 for r in _run_inference(model, tokenizer, list(test_dataset)[:total], agent_id, "eval")
        if r["success"]
    )
    mean_return = correct / total
    print(f"[{agent_id}] Test accuracy: {correct}/{total} ({mean_return:.1%})")

    ckpt_path = f"{results_dir}/{agent_id}/checkpoint"
    trainer.save_model(ckpt_path)
    print(f"[{agent_id}] Checkpoint saved: {ckpt_path}")

    import wandb
    wandb_artifact_name = ""
    if wandb.run is not None:
        try:
            art_name = f"grpo-lora-{agent_id}"
            art = wandb.Artifact(art_name, type="model")
            art.add_dir(ckpt_path)
            wandb.log_artifact(art)
            wandb_artifact_name = art_name
            print(f"[{agent_id}] LoRA adapter uploaded to W&B as '{art_name}'")
        except Exception as e:
            print(f"[{agent_id}] W&B artifact upload failed (non-fatal): {e}")

    weave_run_id = ""
    try:
        call = weave.get_current_call()
        if call is not None:
            weave_run_id = str(call.id)
    except Exception:
        pass

    result = {
        "agent_id": agent_id,
        "algo": "GRPO",
        "env": "Countdown",
        "status": "completed",
        "mean_return": mean_return,
        "std_return": 0.0,
        "steps_trained": step,
        "wall_time_s": time.time() - start,
        "weave_run_id": weave_run_id,
        "checkpoint_path": ckpt_path,
        "wandb_artifact": wandb_artifact_name,
    }
    with open(f"{results_dir}/{agent_id}/eval_result.json", "w") as f:
        json.dump(result, f, indent=2)

    hb.stop("completed")
    print(f"[{agent_id}] done: mean_return={mean_return:.3f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--env-id", default="Countdown")  # accepted but unused
    parser.add_argument("--time-budget", type=int, default=3600)  # 60 min
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--results-dir", default="/workspace/results")
    parser.add_argument("--device", default=os.environ.get("AUTORL_GRPO_DEVICE", "auto"))
    args = parser.parse_args()

    init_weave(args.agent_id)
    train_grpo(
        agent_id=args.agent_id,
        time_budget=args.time_budget,
        lr=args.lr,
        seed=args.seed,
        num_generations=args.num_generations,
        temperature=args.temperature,
        results_dir=args.results_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
