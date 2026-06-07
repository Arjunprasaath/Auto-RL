# Person A — AI Build Guide
## Classic RL Training · Orchestration · CopilotKit UI · Doom Loop Sentinel · Evaluation

> **Your AI builds:** PPO training script · SAC training script · MuJoCo video render · Orchestrator agent · Sentinel agent · Evaluator agent · asyncio swarm runner · CopilotKit frontend (Next.js) · Weave integration · Pydantic schemas

> **You do NOT build:** A2C training script · Countdown environment · GRPO training · Countdown inference · Race Dashboard · Model Viewer. Person B owns these.

> **Interface with Person B:** You write `spawn_plan.json`. Person B reads it. Person B writes `eval_result.json` + `heartbeat.json`. You read them. That is the only dependency.

---

## Phase 0 — Hour 0: Schema Lock & Repo Scaffold

### 0.1 Create Repo Structure

```bash
mkdir -p autorl/{orchestrator,agents,evaluator,training,environments,runpod,model_viewer}
mkdir -p autorl/ui
touch autorl/{orchestrator,agents,evaluator}/__init__.py
cd autorl && git init && git checkout -b person-a
```

### 0.2 Create Pydantic Schemas — `orchestrator/schemas.py`

This is the contract between Person A and Person B. Write it first, commit it, push to main.

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime

class SpawnPlanEntry(BaseModel):
    id: str                                    # "agent_1"
    algo: str                                  # "PPO", "SAC", "A2C", "GRPO"
    env: str                                   # "HalfCheetah-v5", "Countdown"
    exec: Literal["local", "runpod"]
    time_budget_min: int                       # 10 for MuJoCo, 20 for Countdown
    hparams: dict = Field(default_factory=dict)

class Heartbeat(BaseModel):
    agent_id: str
    timestamp: datetime
    status: Literal["starting","training","completed","failed","restarted"]
    steps_completed: int = 0
    current_reward: float = 0.0
    loss: Optional[float] = None
    anomaly: Optional[str] = None             # "nan_loss", "plateau", None

class EvalResult(BaseModel):
    agent_id: str
    algo: str
    env: str
    status: Literal["completed","failed","timed_out","restarted"]
    mean_return: float = 0.0
    std_return: float = 0.0
    steps_trained: int = 0
    wall_time_s: float = 0.0
    weave_run_id: str = ""
    checkpoint_path: str = ""
```

> **CRITICAL:** Commit this file and push to main before splitting. Person B pulls it immediately and imports these models in their training scripts.

### 0.3 Write SCHEMA.md

Copy the Pydantic definitions and the three JSON examples (from the project doc) into `SCHEMA.md`. This is the human-readable contract. Both people can refer to it without reading code.

### 0.4 Create `pyproject.toml`

```toml
[project]
name = "autorl"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "weave",
    "anthropic",
    "pydantic>=2.0",
    "fastapi",
    "uvicorn",
    "copilotkit",
    "stable-baselines3[extra]",
    "gymnasium[mujoco]",
    "imageio[ffmpeg]",
    "torch",
]
```

### 0.5 Create `.env.template`

```
WANDB_API_KEY=
ANTHROPIC_API_KEY=
RUNPOD_API_KEY=
WEAVE_PROJECT=autorl
```

---

## Phase 1 — Hours 0–4: RL Training Scripts (ML Work)

### 1.1 `training/train_ppo.py` — PPO Training Script

Person A owns the PPO and SAC scripts that run locally on MuJoCo. Build PPO first — SAC derives from it.

```python
import argparse, json, time, os, weave
from stable_baselines3 import PPO
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
    model = PPO("MlpPolicy", args.env_id,
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

        # Check for Sentinel nudge
        nudge = hb.check_nudge()
        if nudge:
            new_lr = nudge.get("lr", args.lr)
            model.policy.optimizer.param_groups[0]["lr"] = new_lr
            print(f"[{args.agent_id}] Nudged: lr={new_lr}")

    mean_r, std_r = evaluate_policy(model, model.get_env(), n_eval_episodes=20)

    ckpt = f"{args.results_dir}/{args.agent_id}/model.zip"
    model.save(ckpt)

    result = {
        "agent_id": args.agent_id, "algo": "PPO",
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

> **NaN HANDLING:** When `--lr 1.0` (the deliberately bad agent), training produces NaN loss within ~100 steps. The heartbeat writer detects this and sets `anomaly="nan_loss"`. The Sentinel reads this and kills the agent. The training script does NOT need to handle NaN — the Sentinel handles it.

### 1.2 `training/train_sac.py` — SAC Training Script

Derive from `train_ppo.py` — three changes only: import, model class, algo name in result dict. SAC is off-policy and typically outperforms PPO on MuJoCo locomotion in the same time budget, so it should win the local race.

```python
from stable_baselines3 import SAC

# Same CLI args and loop as train_ppo.py.
# Key difference: SAC uses a replay buffer.
model = SAC("MlpPolicy", args.env_id,
            learning_rate=args.lr,
            buffer_size=100_000,
            learning_starts=1000,   # reward=0 for first ~1000 steps — expected
            seed=args.seed, verbose=0)

# In the result dict: "algo": "SAC"
```

Verify locally before integration:
```bash
python training/train_sac.py --agent-id test_sac --env-id HalfCheetah-v5 --time-budget 120 --lr 3e-4
# Expect mean_return ~500-1500 after 2 min (needs replay warmup)
```

### 1.3 `model_viewer/render_mujoco.py` — MuJoCo Video Render

After the Evaluator picks the best MuJoCo agent, the Orchestrator calls this script. The video path is sent to Person B's ModelViewer component.

```python
import gymnasium, imageio, argparse
from stable_baselines3 import PPO, SAC, A2C

ALGO_MAP = {"PPO": PPO, "SAC": SAC, "A2C": A2C}

def render_video(checkpoint_path: str, env_id: str, algo: str,
                 output_path: str, n_steps: int = 500):
    env = gymnasium.make(env_id, render_mode="rgb_array")
    model = ALGO_MAP[algo].load(checkpoint_path)

    obs, _ = env.reset()
    frames = []

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            obs, _ = env.reset()

    imageio.mimsave(output_path, frames, fps=30)
    env.close()
    print(f"Video saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", default="HalfCheetah-v5")
    parser.add_argument("--algo", default="SAC")
    parser.add_argument("--output", default="results/best_mujoco.mp4")
    args = parser.parse_args()
    render_video(args.checkpoint, args.env_id, args.algo, args.output)
```

---

## Phase 2 — Hours 4–8: Orchestrator Agent

### 2.1 `orchestrator/orchestrator_agent.py`

The Orchestrator takes a user prompt, decides environment family, selects algorithms, and emits `spawn_plan.json`. An LLM call makes these decisions — this is what makes it agentic.

**What the AI must build:**

- A function decorated with `@weave.op` that takes a user prompt string and returns a list of `SpawnPlanEntry` objects
- Calls Claude Haiku with a system prompt explaining available environments and algorithms
- Parses LLM output as JSON, validates each entry through `SpawnPlanEntry` (Pydantic)
- On validation failure: retry once with the error message appended. On second failure: return hard-coded default plan
- Always includes one agent with `hparams: {"lr": 1.0}` to guarantee Sentinel activation during demo

**System prompt for the Orchestrator LLM call:**

```
You are the AutoRL Orchestrator. Given a user task description, decide which RL experiments to run.

Available environments:
- MuJoCo (exec: local): HalfCheetah-v5, Hopper-v5
  Algorithms: PPO, SAC, A2C (Stable-Baselines3)
  Time budget: 10 minutes per agent
- Countdown arithmetic puzzle (exec: runpod):
  Task: use given numbers with +,-,*,/ to reach a target number
  Algorithms: GRPO (no SFT required, Qwen2.5-3B-Instruct)
  Time budget: 20 minutes per agent

Rules:
- Output valid JSON array matching SpawnPlanEntry schema
- Use different seeds for same-algo agents
- Include exactly one agent with lr=1.0 to test fault tolerance
- Default to N=4 unless user specifies otherwise

Output ONLY the JSON array, no other text.
```

**Validation loop:**

```python
@weave.op(name="Orchestrator")
async def create_spawn_plan(task: str) -> list[SpawnPlanEntry]:
    for attempt in range(2):
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": task}]
        )
        try:
            raw = json.loads(response.content[0].text)
            plan = [SpawnPlanEntry(**entry) for entry in raw]
            with open("spawn_plan.json", "w") as f:
                json.dump([e.model_dump() for e in plan], f)
            return plan
        except Exception as e:
            task = f"{task}\n\nPrevious attempt failed: {e}. Fix the JSON."
    return default_plan()  # hard-coded fallback
```

### 2.2 `orchestrator/swarm_runner.py`

The asyncio event loop that spawns all agents in parallel.

```python
import asyncio, weave

@weave.op(name="SwarmRunner")
async def run_swarm(plan: list[SpawnPlanEntry]) -> list[EvalResult]:
    stop_event = asyncio.Event()
    sentinel_task = asyncio.create_task(
        run_sentinel(agent_ids=[e.id for e in plan], stop_event=stop_event)
    )

    training_tasks = [
        asyncio.create_task(run_training_agent(entry))
        for entry in plan
    ]

    max_budget = max(e.time_budget_min for e in plan) + 2  # 2 min grace
    await asyncio.wait(training_tasks, timeout=max_budget * 60)

    stop_event.set()
    await sentinel_task

    results = []
    for entry in plan:
        result_path = f"results/{entry.id}/eval_result.json"
        if os.path.exists(result_path):
            with open(result_path) as f:
                results.append(EvalResult(**json.load(f)))
    return results
```

### 2.3 `agents/training_agent.py`

Wraps training scripts as asyncio subprocesses. For PPO and SAC (local, your scripts), launch directly. For A2C (local, Person B's script) and GRPO (RunPod, Person B's script), same pattern — the wrapper doesn't care who wrote the script.

```python
@weave.op(name="TrainingAgent_{entry.id}_{entry.algo}")
async def run_training_agent(entry: SpawnPlanEntry):
    os.makedirs(f"results/{entry.id}", exist_ok=True)

    if entry.exec == "local":
        script = f"training/train_{entry.algo.lower()}.py"
        cmd = [
            "python", script,
            "--agent-id", entry.id,
            "--env-id", entry.env,
            "--time-budget", str(entry.time_budget_min * 60),
            "--lr", str(entry.hparams.get("lr", 3e-4)),
            "--seed", str(entry.hparams.get("seed", 42)),
            "--results-dir", "results",
        ]
        process = await asyncio.create_subprocess_exec(*cmd)
        await process.wait()

    elif entry.exec == "runpod":
        from runpod.pod_manager import ssh_exec
        cmd = (
            f"python /workspace/training/train_grpo_countdown.py "
            f"--agent-id {entry.id} --time-budget {entry.time_budget_min * 60} "
            f"--seed {entry.hparams.get('seed', 42)} --lr {entry.hparams.get('lr', 1e-6)}"
        )
        ssh_exec(POD_ID, cmd)
```

---

## Phase 3 — Hours 8–14: Sentinel & CopilotKit Scaffold

### 3.1 `agents/sentinel.py` — Doom Loop Sentinel (LLM-based)

> **Implementation status: COMPLETE.** The sentinel is fully implemented as an LLM-based agent in `agents/sentinel.py`. `agents/training_agent.py` lives alongside it. `orchestrator/swarm_runner.py` imports both from `agents.*`.

**Detection** is rule-based (fast, reliable): read `heartbeat.json` every 30 s.
**Intervention** is LLM-driven: GPT receives the failed agent's config, the failure reason, and the full history of prior interventions on this run, then suggests a new hyperparameter configuration. The sentinel kills the agent and relaunches it with the LLM-suggested config.
**Memory**: every intervention is appended to `sentinel_log.json` in the run directory so the history of "what was tried and what happened" persists after the run.

```python
# agents/sentinel.py — key structure (simplified)
from agents import Agent, AgentOutputSchema, Runner  # openai-agents SDK
from pydantic import BaseModel

class SentinelHparams(BaseModel):
    lr: float
    seed: int
    n_steps: int | None = None
    ent_coef: float | None = None
    gamma: float | None = None

_sentinel_agent = Agent(
    name="DoomLoopSentinelLLM",
    instructions=SENTINEL_SYSTEM_PROMPT,   # rules: never lr>=0.1, never repeat failed lr, etc.
    model=OPENAI_MODEL,
    output_type=AgentOutputSchema(SentinelHparams, strict_json_schema=False),
)

@weave.op(name="SentinelLLM")
async def _llm_suggest_hparams(entry_dict, failure_reason, prior_interventions) -> dict:
    prompt = f"Agent failed: {failure_reason}\nOriginal config: {entry_dict}\nPrior interventions: {prior_interventions}"
    result = await Runner.run(_sentinel_agent, prompt)
    return result.final_output.model_dump(exclude_none=True)

@weave.op(name="DoomLoopSentinel")
async def run_sentinel(agent_ids, results_dir, stop_event):
    # Detection loop — rule-based, every 30 s
    while not stop_event.is_set():
        for agent_id in agent_ids:
            hb = read_heartbeat(...)
            if hb["anomaly"] == "nan_loss":
                new_hparams = await _llm_suggest_hparams(entry, "nan_loss", prior_log)
                log_intervention(...)       # writes to sentinel_log.json
                await kill_training_agent(agent_id)
                asyncio.create_task(run_training_agent(entry, hparams_override=new_hparams))
            elif age_s > 120 and not nudged:
                new_hparams = await _llm_suggest_hparams(entry, "stale_heartbeat_nudge", prior_log)
                write_nudge(agent_id, new_hparams)  # agent picks this up via hb.check_nudge()
            elif age_s > 240 and nudged and not restarted:
                new_hparams = await _llm_suggest_hparams(entry, "stale_after_nudge", prior_log)
                await kill_and_restart(entry, new_hparams)
        await asyncio.sleep(30)
    await _check_all()  # final sweep before shutdown
```

**`agents/__init__.py` — SDK bootstrap:** Because the local `agents/` directory shadows the installed `openai-agents` SDK (which also uses the module name `agents`), the `__init__.py` contains a bootstrap that temporarily registers the SDK as `agents` in `sys.modules` while it executes its own init, then restores this package. This makes `from agents import Agent, AgentOutputSchema, Runner` work project-wide.

**Nudge implementation:** Write `results/{agent_id}/nudge.json` with the LLM-suggested hparams dict. The training script picks this up via `hb.check_nudge()` on every 60 s tick, applies the new hparams, and deletes the file.

**Kill + restart:** `agents/training_agent.kill_training_agent(agent_id)` — SIGTERM → SIGKILL the subprocess via the `PROCESSES` dict. Then `run_training_agent(entry, hparams_override=new_hparams)` relaunches it. The outcome (completed / failed_again) is written back into `sentinel_log.json` when the restarted agent finishes.

**`sentinel_log.json` example entry:**
```json
{
  "timestamp": "2026-06-07T00:44:28Z",
  "agent_id": "agent_4",
  "failure_reason": "nan_loss",
  "failed_hparams": {"lr": 1.0, "seed": 42},
  "heartbeat_at_failure": {"steps_completed": 70000, "anomaly": "nan_loss"},
  "llm_suggested_hparams": {"lr": 0.0003, "seed": 1337, "n_steps": 2048, "ent_coef": 0.01},
  "outcome": "completed"
}
```

### 3.2 CopilotKit Next.js App

```bash
cd autorl/ui
npx create-next-app@latest . --typescript --tailwind --app --src-dir
npm install @copilotkit/react-core @copilotkit/react-ui
pip install copilotkit fastapi uvicorn
```

**`ui/agent/middleware.py` — AG-UI bridge:**

```python
from fastapi import FastAPI
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitSDK, Action

app = FastAPI()

async def run_autorl(task: str):
    # 1. Call orchestrator_agent → spawn_plan
    # 2. Yield state updates for CopilotKit (agent cards)
    # 3. Run swarm_runner
    # 4. Yield results to UI
    pass

sdk = CopilotKitSDK(actions=[
    Action(name="run_autorl", handler=run_autorl,
           description="Run the AutoRL training race given a task description")
])
add_fastapi_endpoint(app, sdk, "/copilotkit")
```

> **NOTE:** Check `docs.copilotkit.ai` for the latest FastAPI integration pattern before building. The AG-UI SDK API can change between minor versions.

**`ui/app/page.tsx` — Main page:**

```typescript
"use client";
import { CopilotChat } from "@copilotkit/react-ui";
import { useCopilotAction } from "@copilotkit/react-core";

export default function Home() {
  useCopilotAction({
    name: "approve_spawn",
    description: "Approve spawning N training agents",
    parameters: [
      { name: "plan_summary", type: "string" },
      { name: "estimated_cost", type: "string" },
    ],
    renderAndWaitForResponse: ({ args }) => (
      <ApprovalCard summary={args.plan_summary} cost={args.estimated_cost} />
    ),
  });

  return <CopilotChat />;
}
```

### 3.3 `ui/components/ApprovalCard.tsx` + `SentinelAlert.tsx`

**ApprovalCard:** rendered by `useCopilotAction` when the Orchestrator proposes a spawn plan. Shows algo list, env targets, estimated cost, Approve/Reject buttons.

**SentinelAlert.tsx:**

```typescript
// Renders as a warning card in the CopilotKit chat when Sentinel fires:
// "⚠ Agent 4 (PPO lr=1.0) — NaN loss detected. Killing and restarting with lr=3e-4."
// Use useCopilotAction with name "sentinel_alert"
```

> **Race Dashboard and Model Viewer are built by Person B.** Your job is to push the right state from the swarm runner so those components have data to render. After every heartbeat read cycle, push agent status updates to CopilotKit state. After the Evaluator returns, send the video path and countdown_solve.json path as state updates.

---

## Phase 4 — Hours 14–28: Weave, Evaluator & Pipeline Entry Point

### 4.1 `orchestrator/main.py` — Pipeline Entry Point

```python
import weave, asyncio, json, os
from orchestrator.orchestrator_agent import create_spawn_plan
from orchestrator.swarm_runner import run_swarm
from evaluator.evaluator_agent import evaluate_results

weave.init("autorl")

@weave.op(name="AutoRL_Pipeline")
async def main(task: str):
    plan = await create_spawn_plan(task)
    results = await run_swarm(plan)
    rankings = await evaluate_results(results)
    return rankings

if __name__ == "__main__":
    task = input("Describe your RL task: ")
    asyncio.run(main(task))
```

### 4.2 Checkpoint — Hour 10: Weave Trace Review

Run the pipeline with a mock training script that writes dummy `eval_result.json` after 10 seconds. Verify in Weave:

- `AutoRL_Pipeline` → `Orchestrator` → `SwarmRunner` → `TrainingAgent_agent_1`, `TrainingAgent_agent_2`, `DoomLoopSentinel` → `Evaluator` all appear as named nodes
- If the trace looks flat (no hierarchy), check that `@weave.op` calls are nested inside parent `@weave.op` calls

### 4.3 `evaluator/evaluator_agent.py`

```python
@weave.op(name="Evaluator")
async def evaluate_results(results: list[EvalResult]) -> dict:
    mujoco_results = [r for r in results if r.env != "Countdown"]
    countdown_results = [r for r in results if r.env == "Countdown"]

    rankings = {}

    for group_name, group in [("MuJoCo", mujoco_results), ("Countdown", countdown_results)]:
        if not group:
            continue

        prompt = f"""Rank these RL training results for {group_name}:

{json.dumps([r.model_dump() for r in group], indent=2)}

Consider: mean return (higher is better), stability (lower std),
sample efficiency, and whether the agent completed or failed.
Note any Sentinel interventions (status=restarted).

Output JSON: [{{"rank": 1, "agent_id": "...", "algo": "...", "rationale": "..."}}]"""

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": prompt}]
        )
        rankings[group_name] = json.loads(response.content[0].text)

    _push_weave_evaluation(results, rankings)

    with open("rankings.json", "w") as f:
        json.dump(rankings, f)

    return rankings
```

### 4.4 Weave Evaluation Integration

```python
class ReturnScorer(weave.Scorer):
    @weave.op
    def score(self, output, expected=None):
        return {
            "return": output.get("mean_return", 0),
            "stability": 1.0 / max(output.get("std_return", 1), 0.01)
        }

evaluation = weave.Evaluation(
    dataset=results_as_dataset,
    scorers=[ReturnScorer()]
)
await evaluation.evaluate(evaluator_fn)
```

### 4.5 `evaluator/reporter.py`

Template fill — no LLM call needed. Format `rankings.json` into `run_report.md`:

```markdown
# AutoRL Run Report
**Task:** {task_description}
**Weave project:** {weave_url}

## MuJoCo Results
1. SAC — HalfCheetah-v5 — mean_return: 5231 — {rationale}
2. PPO — HalfCheetah-v5 — mean_return: 3012 — {rationale}

## Countdown Results
1. GRPO seed=42 — mean_return: 0.67 — {rationale}
2. GRPO seed=123 (restarted by Sentinel) — mean_return: 0.61 — {rationale}

## Sentinel Interventions
- Agent 4 (PPO lr=1.0): NaN loss at step 45 → killed + restarted with lr=3e-4
```

### 4.6 Hour 16 Hard Gate

> **STOP AND VERIFY:** PPO + SAC on HalfCheetah, both local, through the complete pipeline: Orchestrator → spawn → Training Agents → heartbeats → Sentinel monitoring → Evaluator → rankings → CopilotKit UI showing results. If this is not green by Hour 16, do not touch RunPod until it is.

### 4.7 Model-in-Action Trigger

After Evaluator returns, the Orchestrator:

1. Identifies the best MuJoCo checkpoint from `rankings.json`
2. Calls `subprocess.run(["python", "model_viewer/render_mujoco.py", "--checkpoint", path, "--algo", algo, "--output", "results/best_mujoco.mp4"])` — your script
3. Identifies the best Countdown checkpoint
4. Calls `subprocess.run(["python", "model_viewer/countdown_inference.py", "--checkpoint", path, "--output", "results/countdown_solve.json"])` — Person B's script
5. Sends both file paths to CopilotKit state for Person B's `ModelViewer` component to render

---

## Phase 5 — Hours 28–36: Demo Prep & Submission

### 5.1 Sentinel Demo Verification

Confirm that the `lr=1.0` agent triggers the Sentinel during every run. The heartbeat should show `anomaly: "nan_loss"` within 30–60 seconds. Verify the SentinelAlert card appears in CopilotKit.

### 5.2 Set Weave Project to Public

In the W&B dashboard: Settings → Privacy → set to Public. Copy the Weave project URL into `run_report.md` and the submission form.

### 5.3 Demo Dry Run (Hour 30, with Person B)

- Run full pipeline from cold start
- Time it — should complete in under 15 minutes
- Verify all beats: approval card → race dashboard → Sentinel alert → leaderboard → model in action
- Record demo video immediately after a successful run

### 5.4 README.md

Already written at the repo root. Update the Weave project URL and demo video link after Hour 32.

---

*Person A Build Guide — AutoRL Hackathon*
