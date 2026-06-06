# Person B — AI Build Guide
## A2C Training · Countdown Environment · GRPO · RunPod Compute · Race Dashboard · Model Viewer

> **Your AI builds:** A2C training script · heartbeat writer · Weave callback · RunPod pod manager · Countdown puzzle environment · GRPO training script (no SFT) · Countdown live solve output · Race Dashboard · Model Viewer

> **You do NOT build:** PPO training script · SAC training script · MuJoCo video render · Orchestrator · Sentinel · Evaluator · CopilotKit scaffold. Person A owns these.

> **Interface with Person A:** Person A writes `spawn_plan.json`. Your scripts read it via CLI args. Your scripts write `eval_result.json` + `heartbeat.json`. Person A reads them. That is the only dependency.

---

## Phase 0 — Hour 0: Schema Lock & Environment Setup

### 0.1 Pull Person A's Schema

Person A creates the repo with `schemas.py` and `SCHEMA.md`. Pull main and branch.

```bash
git pull origin main
git checkout -b person-b
```

### 0.2 Install Local Dependencies (Mac M3)

```bash
pip install "stable-baselines3[extra]" gymnasium[mujoco] weave imageio[ffmpeg] pydantic
# Verify MuJoCo works:
python -c "import gymnasium; e=gymnasium.make('HalfCheetah-v5'); e.reset(); print('OK')"
```

### 0.3 Pre-warm RunPod — Do This First (Hours 0–2)

Create a pod immediately. Do not wait until you need GRPO.

```python
# runpod/pod_manager.py
import runpod, subprocess, time

POD_ID = None  # set after create_pod

def create_training_pod(name="autorl-countdown"):
    pod = runpod.create_pod(
        name=name,
        image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        gpu_type_id="NVIDIA GeForce RTX 4090",  # ~$0.44/hr
        gpu_count=1,
        volume_in_gb=30,
        ports="22/tcp",
    )
    global POD_ID
    POD_ID = pod["id"]
    return POD_ID

def get_pod_ssh_info(pod_id):
    """Returns (host, port) for SSH access."""
    pod = runpod.get_pod(pod_id)
    for p in pod["runtime"]["ports"]:
        if p["privatePort"] == 22:
            return pod["runtime"]["ip"], p["publicPort"]

def ssh_exec(pod_id, command, timeout=7200):
    """SSH into pod and run command. Returns stdout."""
    host, port = get_pod_ssh_info(pod_id)
    result = subprocess.run(
        ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no",
         f"root@{host}", command],
        capture_output=True, text=True, timeout=timeout
    )
    return result.stdout

def terminate_pod(pod_id):
    runpod.terminate_pod(pod_id)
```

> **COST:** RTX 4090 is ~$0.44/hr. 20 hours = ~$9. Keep it alive until submission is confirmed. Do NOT terminate early.

### 0.4 Install Dependencies on Pod

```bash
# SSH in and run:
pip install trl transformers datasets accelerate peft bitsandbytes weave pydantic
```

---

## Phase 1 — Hours 0–4: Heartbeat Writer & Weave Callback

### 1.1 `training/callbacks/heartbeat_writer.py` — Build This First

Every training script imports this. It runs as a background thread and is what makes the Sentinel possible. Person A's PPO and SAC scripts also import this — build it before anything else.

```python
import json, threading, time, os
from datetime import datetime, timezone
from pathlib import Path

class HeartbeatWriter:
    def __init__(self, agent_id: str, results_dir: str = "./results"):
        self.agent_id = agent_id
        self.dir = Path(results_dir) / agent_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hb_path = self.dir / "heartbeat.json"
        self.nudge_path = self.dir / "nudge.json"

        self.steps = 0
        self.reward = 0.0
        self.loss = None
        self.anomaly = None
        self.status = "starting"

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.status = "training"
        self._thread.start()

    def update(self, steps: int, reward: float, loss: float = None):
        """Call this from the training loop after each chunk."""
        self.steps = steps
        self.reward = reward
        self.loss = loss
        if loss is not None and (loss != loss or abs(loss) > 1e6):
            self.anomaly = "nan_loss"

    def stop(self, status: str = "completed"):
        self.status = status
        self._write()  # final write
        self._stop.set()

    def check_nudge(self) -> dict | None:
        """Returns new hparams dict if Sentinel wrote a nudge, else None."""
        if self.nudge_path.exists():
            with open(self.nudge_path) as f:
                nudge = json.load(f)
            self.nudge_path.unlink()
            return nudge
        return None

    def _loop(self):
        while not self._stop.is_set():
            self._write()
            self._stop.wait(60)

    def _write(self):
        data = {
            "agent_id": self.agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": self.status,
            "steps_completed": self.steps,
            "current_reward": self.reward,
            "loss": self.loss,
            "anomaly": self.anomaly,
        }
        with open(self.hb_path, "w") as f:
            json.dump(data, f)
```

### 1.2 `training/callbacks/weave_callback.py` — Weave SB3 Callback

```python
import weave
from stable_baselines3.common.callbacks import BaseCallback

class WeaveLogCallback(BaseCallback):
    def __init__(self, agent_id: str, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.agent_id = agent_id
        self.log_freq = log_freq
        self.ep_returns = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_returns.append(info["episode"]["r"])

        if self.num_timesteps % self.log_freq == 0 and self.ep_returns:
            mean_r = sum(self.ep_returns[-10:]) / min(len(self.ep_returns), 10)
            print(f"[{self.agent_id}] step={self.num_timesteps} return={mean_r:.1f}")
        return True
```

---

## Phase 2 — Hours 2–6: A2C Training Script (ML Work)

### 2.1 `training/train_a2c.py` — A2C Training Script

Person B owns A2C — the third MuJoCo racer alongside Person A's PPO and SAC. A2C is synchronous actor-critic and typically lands between PPO and SAC on locomotion tasks.

```python
import argparse, json, time, os, weave
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from training.callbacks.heartbeat_writer import HeartbeatWriter
from training.callbacks.weave_callback import WeaveLogCallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--env-id", default="HalfCheetah-v5")
    parser.add_argument("--time-budget", type=int, default=600)  # seconds
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="./results")
    args = parser.parse_args()

    os.makedirs(f"{args.results_dir}/{args.agent_id}", exist_ok=True)

    hb = HeartbeatWriter(args.agent_id, args.results_dir)
    hb.start()

    weave.init("autorl")
    # A2C: synchronous actor-critic, no replay buffer, trains on-policy
    model = A2C("MlpPolicy", args.env_id,
                learning_rate=args.lr, seed=args.seed, verbose=0)
    cb = WeaveLogCallback(args.agent_id)

    start = time.time()
    total_steps = 0
    CHUNK = 5000

    while time.time() - start < args.time_budget:
        model.learn(total_timesteps=CHUNK, callback=cb, reset_num_timesteps=False)
        total_steps += CHUNK

        last_r = cb.ep_returns[-1] if cb.ep_returns else 0.0
        hb.update(total_steps, last_r, loss=None)

        nudge = hb.check_nudge()
        if nudge:
            new_lr = nudge.get("lr", args.lr)
            model.policy.optimizer.param_groups[0]["lr"] = new_lr
            print(f"[{args.agent_id}] Nudged: lr={new_lr}")

    mean_r, std_r = evaluate_policy(model, model.get_env(), n_eval_episodes=20)

    ckpt = f"{args.results_dir}/{args.agent_id}/model.zip"
    model.save(ckpt)

    result = {
        "agent_id": args.agent_id, "algo": "A2C",
        "env": args.env_id, "status": "completed",
        "mean_return": float(mean_r), "std_return": float(std_r),
        "steps_trained": total_steps,
        "wall_time_s": time.time() - start,
        "weave_run_id": "", "checkpoint_path": ckpt,
    }
    with open(f"{args.results_dir}/{args.agent_id}/eval_result.json", "w") as f:
        json.dump(result, f)

    hb.stop("completed")

if __name__ == "__main__":
    main()
```

Verify locally:
```bash
python training/train_a2c.py --agent-id test_a2c --env-id HalfCheetah-v5 --time-budget 120 --lr 3e-4
# Expected mean_return ~1000-3000 after 2 min — below SAC, sometimes below PPO. That's fine.
```

---

## Phase 3 — Hours 4–8: Countdown Environment & GRPO Setup

### 3.1 `environments/countdown_env.py` — Puzzle Generation & Reward

The Countdown puzzle: given a list of numbers, use `+, -, *, /` to reach a target. Numbers must each be used exactly once.

```python
import random, itertools, operator
from datasets import load_dataset

OPS = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}

def generate_countdown_prompt(numbers: list[int], target: int) -> str:
    return (
        f"Using the numbers {numbers} with operations +, -, *, / "
        f"(each number used exactly once), reach the target: {target}\n"
        f"Show your work step by step, then write the final expression."
    )

def evaluate_solution(model_output: str, target: int, numbers: list[int]) -> float:
    """
    Returns 1.0 (correct), 0.5 (valid expression but wrong result), or 0.0 (invalid).
    """
    import re
    expressions = re.findall(r'[\d\s\+\-\*\/\(\)]+', model_output)

    for expr in reversed(expressions):
        try:
            result = eval(expr.strip())  # safe for controlled input
            if abs(result - target) < 1e-6:
                return 1.0
            else:
                return 0.5
        except:
            continue
    return 0.0

def load_countdown_dataset():
    """Load the public Countdown dataset from HuggingFace."""
    dataset = load_dataset("zouxuhong/Countdown-Tasks-3to4", split="train")
    return dataset
```

> **Why `eval()` is safe here:** The model's output is parsed by regex to extract only digit/operator patterns before calling `eval()`. The training environment is sandboxed on RunPod. Do not use this in production.

### 3.2 `training/train_grpo_countdown.py` — GRPO Cold-Start (No SFT)

Key facts:
- **No SFT required.** Qwen2.5-3B-Instruct already understands the format
- Uses LoRA for memory efficiency (~10GB VRAM on RTX 4090)
- Uses `zouxuhong/Countdown-Tasks-3to4` from HuggingFace — one line to load
- Trains for a fixed time budget (20 min), not a fixed number of steps
- Two agents run with different seeds → different training trajectories

```python
import argparse, json, time, os, weave
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOTrainer, GRPOConfig
from training.callbacks.heartbeat_writer import HeartbeatWriter
from environments.countdown_env import generate_countdown_prompt, evaluate_solution

def countdown_reward_fn(completions: list[str], prompts: list[str],
                         targets: list[int], numbers: list[list[int]], **kwargs) -> list[float]:
    rewards = []
    for completion, target, nums in zip(completions, targets, numbers):
        r = evaluate_solution(completion, target, nums)
        rewards.append(r)
    return rewards

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--time-budget", type=int, default=1200)  # 20 min
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="/workspace/results")
    args = parser.parse_args()

    os.makedirs(f"{args.results_dir}/{args.agent_id}", exist_ok=True)

    hb = HeartbeatWriter(args.agent_id, args.results_dir)
    hb.start()
    weave.init("autorl")

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("zouxuhong/Countdown-Tasks-3to4", split="train")
    dataset = dataset.shuffle(seed=args.seed)

    def format_row(row):
        return {
            "prompt": generate_countdown_prompt(row["numbers"], row["target"]),
            "target": row["target"],
            "numbers": row["numbers"],
        }
    formatted = dataset.map(format_row)

    grpo_config = GRPOConfig(
        learning_rate=args.lr,
        per_device_train_batch_size=4,
        num_generations=4,
        max_completion_length=256,
        temperature=1.0,
        seed=args.seed,
        output_dir=f"{args.results_dir}/{args.agent_id}",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        tokenizer=tokenizer,
        train_dataset=formatted,
        reward_funcs=countdown_reward_fn,
    )

    start = time.time()
    step = 0

    while time.time() - start < args.time_budget:
        trainer.train()
        step += 1

        recent_rewards = [log.get("reward", 0) for log in trainer.state.log_history[-5:]]
        mean_reward = sum(recent_rewards) / max(len(recent_rewards), 1)
        hb.update(step, mean_reward,
                  loss=trainer.state.log_history[-1].get("loss") if trainer.state.log_history else None)

        nudge = hb.check_nudge()
        if nudge:
            for pg in trainer.optimizer.param_groups:
                pg["lr"] = nudge.get("lr", args.lr)

    # Evaluate on 100 test puzzles
    test_dataset = load_dataset("zouxuhong/Countdown-Tasks-3to4", split="test")
    correct = 0
    total = min(100, len(test_dataset))

    model.eval()
    for row in list(test_dataset)[:total]:
        prompt = generate_countdown_prompt(row["numbers"], row["target"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with __import__("torch").no_grad():
            output = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
        completion = tokenizer.decode(output[0], skip_special_tokens=True)
        if evaluate_solution(completion, row["target"], row["numbers"]) == 1.0:
            correct += 1

    mean_return = correct / total

    ckpt_path = f"{args.results_dir}/{args.agent_id}/checkpoint"
    trainer.save_model(ckpt_path)

    result = {
        "agent_id": args.agent_id, "algo": "GRPO",
        "env": "Countdown", "status": "completed",
        "mean_return": mean_return, "std_return": 0.0,
        "steps_trained": step,
        "wall_time_s": time.time() - start,
        "weave_run_id": "", "checkpoint_path": ckpt_path,
    }
    with open(f"{args.results_dir}/{args.agent_id}/eval_result.json", "w") as f:
        json.dump(result, f)

    hb.stop("completed")

if __name__ == "__main__":
    main()
```

> **Important:** The `GRPOTrainer` API in `trl` changes between versions. Check the trl docs for the exact `reward_funcs` signature your installed version expects. The pattern above matches trl ≥ 0.9.

### 3.3 Record Base Model Baseline — Hour 8

**This is the demo's key moment.** Before any GRPO training, run the base model on 20 test puzzles and record the score.

```bash
python -c "
from transformers import pipeline
from environments.countdown_env import generate_countdown_prompt, evaluate_solution
from datasets import load_dataset

pipe = pipeline('text-generation', model='Qwen/Qwen2.5-3B-Instruct', device=0)
test = list(load_dataset('zouxuhong/Countdown-Tasks-3to4', split='test'))[:20]
correct = 0
for row in test:
    prompt = generate_countdown_prompt(row['numbers'], row['target'])
    out = pipe(prompt, max_new_tokens=200, temperature=0.1)[0]['generated_text']
    if evaluate_solution(out, row['target'], row['numbers']) == 1.0:
        correct += 1
print(f'Base model success: {correct}/20 ({correct*5}%)')
"
```

Screenshot the output. This number is your **before** picture for the demo.

---

## Phase 4 — Hours 8–14: Integration & Verification

### 4.1 Integration Test Checklist

Run each command independently and verify the output before Person A connects anything:

```bash
# Test 1: A2C — should produce eval_result.json + heartbeat.json in 2 min
python training/train_a2c.py \
  --agent-id test_a2c --env-id HalfCheetah-v5 \
  --time-budget 120 --lr 3e-4 --seed 42 --results-dir ./test_results
cat test_results/test_a2c/eval_result.json
cat test_results/test_a2c/heartbeat.json

# Test 2: GRPO on RunPod (SSH)
ssh_exec(POD_ID, "python /workspace/training/train_grpo_countdown.py "
                 "--agent-id test_grpo --time-budget 300 --seed 42 "
                 "--results-dir /workspace/results")
# scp /workspace/results/test_grpo/eval_result.json to verify
```

### 4.2 Hour 16 Gate — Provide to Person A

By Hour 16, Person A's orchestrator must be able to launch your scripts. Confirm:

- `train_a2c.py` accepts all CLI args and writes valid `eval_result.json` + `heartbeat.json`
- GRPO SSH execution works from Person A's `training_agent.py` wrapper
- `nudge.json` handling works: write a test nudge file and verify `check_nudge()` picks it up

---

## Phase 5 — Hours 14–22: UI — Race Dashboard & Model Viewer

### 5.1 `ui/components/RaceDashboard.tsx`

Person B builds the Race Dashboard because you know what training outputs look like. Use `useCoAgentStateRender` to subscribe to state updates pushed by Person A's AG-UI bridge.

```typescript
// Each agent card shows:
// - Agent name (e.g. "Agent 2 — SAC — HalfCheetah")
// - Status badge: "starting" (gray) | "training" (blue pulse)
//                 "completed" (green) | "failed/restarted" (red)
// - Progress bar: time elapsed / time budget
// - Current reward: latest heartbeat value, updates every 30s
// - Sentinel alert banner if agent status is "restarted"

// Data source: Person A's swarm runner pushes agent state to CopilotKit
// after every heartbeat read cycle. Use useCoAgentStateRender to subscribe.
```

### 5.2 `ui/components/ModelViewer.tsx`

After evaluation completes, this component shows the best model in action. Two sub-views:

```typescript
// MuJoCo sub-view:
// HTML5 <video> element loading the mp4 path from Person A's render_mujoco.py.
// Hidden until Person A sends the video path via CopilotKit state.

// Countdown sub-view:
// Animated card sequence — one card per puzzle:
//   [4, 7, 2, 9] → 24
//   Model's chain-of-thought revealed line by line (300ms delay per line)
//   ✅ success or ❌ failure at the end
// Data: loads countdown_solve.json written by countdown_inference.py below.
// Animate with setTimeout — reveal each line sequentially.

// Both sub-views are hidden until Person A's Orchestrator
// sends the model-in-action state update via AG-UI.
```

---

## Phase 6 — Hours 22–28: Countdown Inference & Final Integration

### 6.1 `model_viewer/countdown_inference.py` — Live Puzzle Solve

Runs 5 test puzzles and outputs JSON that Person A's CopilotKit `ModelViewer` renders step by step.

```python
import argparse, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from environments.countdown_env import generate_countdown_prompt, evaluate_solution

def run_inference(checkpoint_path: str, n: int = 5, output_path: str = "results/countdown_solve.json"):
    base_model = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()

    test_data = list(load_dataset("zouxuhong/Countdown-Tasks-3to4", split="test"))[:n]

    results = []
    for row in test_data:
        prompt = generate_countdown_prompt(row["numbers"], row["target"])
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=300, temperature=0.1)
        completion = tokenizer.decode(output[0], skip_special_tokens=True)
        response = completion[len(prompt):]

        score = evaluate_solution(response, row["target"], row["numbers"])
        results.append({
            "numbers": row["numbers"],
            "target": row["target"],
            "prompt": prompt,
            "model_response": response,
            "success": score == 1.0,
        })
        print(f"[{row['numbers']} → {row['target']}] success={score == 1.0}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--output", default="results/countdown_solve.json")
    args = parser.parse_args()
    run_inference(args.checkpoint, args.n, args.output)
```

---

## Phase 7 — Hours 28–36: Demo Prep & Final Checks

### 7.1 Record Before/After Screenshots

```bash
# BEFORE (base model, before GRPO):
# Screenshot terminal output: "Base model success: X/20 (X%)"

# AFTER (trained model):
python model_viewer/countdown_inference.py \
  --checkpoint /workspace/results/agent_3/checkpoint \
  --n 20 --output results/countdown_eval_final.json
# Screenshot: how many of 20 did it solve?
```

### 7.2 Validate Expected Numbers

| Agent | Expected result | Notes |
|---|---|---|
| A2C on HalfCheetah (10 min) | mean_return ~1000–3000 | Below SAC and PPO — that's the narrative |
| PPO on HalfCheetah (10 min) | mean_return ~2000–4000 | Person A's script |
| SAC on HalfCheetah (10 min) | mean_return ~3000–6000 | Should win MuJoCo race |
| GRPO seed=42 (20 min) | success rate ~55–67% | Documented baseline for 3B cold-start |
| Bad agent (lr=1.0) | NaN within 60s, Sentinel fires | Intentional — verify in Weave trace |

### 7.3 Contingency: N=2 Local-Only Fallback

If RunPod or GRPO is broken at Hour 28:

- `spawn_plan.json` with only 3 agents: PPO + SAC + A2C on HalfCheetah, all local
- The demo still works: judges see the race, the Sentinel, the Evaluator, and a MuJoCo video
- Countdown is a bonus, not a requirement

### 7.4 Pod Termination

> **Do NOT terminate the RunPod pod until:** the demo video is recorded, the submission is confirmed, and Person A gives the all-clear. Only then run `runpod.terminate_pod(POD_ID)`.

---

*Person B Build Guide — AutoRL Hackathon*
