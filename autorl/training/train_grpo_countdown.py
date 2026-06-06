"""
GRPO cold-start training on the Countdown arithmetic puzzle (Person B).

Runs on RunPod GPU (/workspace/venv/bin/python). No SFT required —
Qwen2.5-3B-Instruct already understands arithmetic format; GRPO trains
multi-step planning from scratch.

Data contract (same as MuJoCo scripts):
  - heartbeat.json written every 60s (via HeartbeatWriter)
  - honours Sentinel nudges (results/{agent_id}/nudge.json)
  - writes eval_result.json on completion

Run from /workspace (repo root on the pod), e.g.:
    /workspace/venv/bin/python training/train_grpo_countdown.py \
        --agent-id agent_4 --time-budget 1200 --lr 1e-6 --seed 42 \
        --results-dir /workspace/results

CLI args match what Person A's training_agent.py wrapper passes:
    --agent-id, --time-budget, --lr, --seed, --results-dir
(Note: --env-id is accepted but ignored; environment is always Countdown)
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
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from environments.countdown_env import evaluate_solution, generate_countdown_prompt, load_countdown_dataset
from training.callbacks.heartbeat_writer import HeartbeatWriter

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def countdown_reward_fn(
    completions: list[str],
    prompts: list[str],
    target: list[int],
    numbers: list[list[int]],
    **kwargs,
) -> list[float]:
    """Reward function passed to GRPOTrainer."""
    return [
        evaluate_solution(completion, t, nums)
        for completion, t, nums in zip(completions, target, numbers)
    ]


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


@weave.op(name="GRPO_Countdown_Training")
def train_grpo(agent_id, time_budget, lr, seed, num_generations, temperature, results_dir):
    """Time-budgeted GRPO training on Countdown. Traced as a Weave op."""
    os.makedirs(f"{results_dir}/{agent_id}", exist_ok=True)

    hb = HeartbeatWriter(agent_id, results_dir)
    hb.start()

    print(f"[{agent_id}] Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto"
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"[{agent_id}] Loading dataset...")
    dataset = load_countdown_dataset(split="train", seed=seed)
    dataset = dataset.shuffle(seed=seed)

    def format_row(row):
        return {
            "prompt": generate_countdown_prompt(row["nums"], row["target"]),
            "target": row["target"],
            "numbers": row["nums"],
        }

    formatted = dataset.map(format_row, remove_columns=dataset.column_names)

    grpo_config = GRPOConfig(
        learning_rate=lr,
        per_device_train_batch_size=4,
        num_generations=num_generations,
        max_completion_length=256,
        temperature=temperature,
        seed=seed,
        output_dir=f"{results_dir}/{agent_id}",
        logging_steps=1,
        save_steps=9999,  # don't auto-save; we save manually at the end
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=formatted,
        reward_funcs=countdown_reward_fn,
    )

    start = time.time()
    step = 0

    print(f"[{agent_id}] Starting GRPO training (budget: {time_budget}s, lr={lr})...")
    while time.time() - start < time_budget:
        trainer.train()
        step += 1

        recent = trainer.state.log_history[-5:] if trainer.state.log_history else []
        mean_reward = sum(log.get("reward", 0) for log in recent) / max(len(recent), 1)
        last_loss = recent[-1].get("loss") if recent else None
        hb.update(step, mean_reward, loss=last_loss)

        nudge = hb.check_nudge()
        if nudge:
            new_lr = nudge.get("lr", lr)
            for pg in trainer.optimizer.param_groups:
                pg["lr"] = new_lr
            print(f"[{agent_id}] Nudged: lr={new_lr}")

    # Evaluate on 100 test puzzles (held-out 5% split)
    print(f"[{agent_id}] Evaluating on test set...")
    test_dataset = load_countdown_dataset(split="test", seed=seed)
    total = min(100, len(test_dataset))
    correct = 0

    model.eval()
    for row in list(test_dataset)[:total]:
        prompt = generate_countdown_prompt(row["nums"], row["target"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=False)
        completion = tokenizer.decode(output[0], skip_special_tokens=True)
        if evaluate_solution(completion, row["target"], row["nums"]) == 1.0:
            correct += 1

    mean_return = correct / total
    print(f"[{agent_id}] Test accuracy: {correct}/{total} ({mean_return:.1%})")

    ckpt_path = f"{results_dir}/{agent_id}/checkpoint"
    trainer.save_model(ckpt_path)
    print(f"[{agent_id}] Checkpoint saved: {ckpt_path}")

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
    parser.add_argument("--time-budget", type=int, default=1200)  # 20 min
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--results-dir", default="/workspace/results")
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
    )


if __name__ == "__main__":
    main()
