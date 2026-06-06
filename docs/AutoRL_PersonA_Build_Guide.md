# Person A — AI Build Guide
## Orchestration · CopilotKit UI · Doom Loop Sentinel · Evaluation

> **Your AI builds:** Orchestrator agent · Sentinel agent · Evaluator agent · asyncio swarm runner · CopilotKit frontend (Next.js) · Weave integration · Pydantic schemas

> **You do NOT build:** Training scripts · RunPod pod manager · Countdown environment · Video render · Countdown inference. Person B owns these.

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

## Phase 1 — Hours 0–4: Orchestrator Agent

### 1.1 `orchestrator/orchestrator_agent.py`

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
- Include exactly one GRPO agent with lr=1.0 to test fault tolerance
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
            # Write to disk
            with open("spawn_plan.json", "w") as f:
                json.dump([e.model_dump() for e in plan], f)
            return plan
        except Exception as e:
            task = f"{task}\n\nPrevious attempt failed: {e}. Fix the JSON."
    return default_plan()  # hard-coded fallback
```

### 1.2 `orchestrator/swarm_runner.py`

The asyncio event loop that spawns all agents in parallel.

**What the AI must build:**

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
    
    # Wait for all training agents (with timeout)
    max_budget = max(e.time_budget_min for e in plan) + 2  # 2 min grace
    await asyncio.wait(training_tasks, timeout=max_budget * 60)
    
    stop_event.set()
    await sentinel_task
    
    # Collect results
    results = []
    for entry in plan:
        result_path = f"results/{entry.id}/eval_result.json"
        if os.path.exists(result_path):
            with open(result_path) as f:
                results.append(EvalResult(**json.load(f)))
    return results
```

### 1.3 `agents/training_agent.py`

Wraps Person B's training scripts as asyncio subprocesses.

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
        # Call Person B's pod_manager to SSH and run the script
        from runpod.pod_manager import ssh_exec
        cmd = f"python /workspace/training/train_grpo_countdown.py " \
              f"--agent-id {entry.id} --time-budget {entry.time_budget_min * 60} " \
              f"--seed {entry.hparams.get('seed', 42)} --lr {entry.hparams.get('lr', 1e-6)}"
        ssh_exec(POD_ID, cmd)
```

> **KEY INSIGHT:** The Training Agent wrapper does NOT contain RL logic. It launches Person B's training scripts as subprocesses. Person A never writes RL code.

---

## Phase 2 — Hours 4–8: Sentinel & CopilotKit Scaffold

### 2.1 `agents/sentinel.py` — Doom Loop Sentinel

```python
import asyncio, json, weave, os
from datetime import datetime, timezone
from orchestrator.schemas import Heartbeat

@weave.op(name="DoomLoopSentinel")
async def run_sentinel(
    agent_ids: list[str],
    results_dir: str = "./results",
    check_interval: int = 30,
    nudge_threshold_s: int = 120,
    kill_threshold_s: int = 240,
    stop_event: asyncio.Event = None,
):
    nudged = {}       # agent_id -> timestamp of nudge
    restarted = set() # agent_ids that have been restarted once
    killed = set()    # agent_ids permanently killed

    while not stop_event.is_set():
        for agent_id in agent_ids:
            if agent_id in killed:
                continue
            
            hb_path = f"{results_dir}/{agent_id}/heartbeat.json"
            if not os.path.exists(hb_path):
                continue
            
            with open(hb_path) as f:
                hb = Heartbeat(**json.load(f))
            
            now = datetime.now(timezone.utc)
            age_s = (now - hb.timestamp).total_seconds()
            
            # NaN: skip nudge, immediate kill+restart
            if hb.anomaly == "nan_loss":
                if agent_id not in restarted:
                    await _kill_and_restart(agent_id, "nan_loss")
                    restarted.add(agent_id)
                else:
                    await _kill_permanently(agent_id)
                    killed.add(agent_id)
                continue
            
            # Stale heartbeat
            if age_s > nudge_threshold_s and agent_id not in nudged:
                await _nudge(agent_id, results_dir)
                nudged[agent_id] = now
            
            elif age_s > kill_threshold_s and agent_id in nudged:
                if agent_id not in restarted:
                    await _kill_and_restart(agent_id, "stale_after_nudge")
                    restarted.add(agent_id)
                else:
                    await _kill_permanently(agent_id)
                    killed.add(agent_id)
        
        await asyncio.sleep(check_interval)
```

**Nudge implementation:** Write `results/{agent_id}/nudge.json` with `{"lr": current_lr / 2, "seed": new_seed}`. Person B's heartbeat writer checks for this file every 60s, applies new hparams, deletes the file.

**Kill + restart:** Put the agent PID in a shared dict. Call `process.terminate()`. Request the swarm runner (via `asyncio.Queue`) to spawn a replacement with `lr=3e-4` and a new seed.

### 2.2 CopilotKit Next.js App

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

---

## Phase 3 — Hours 8–14: Weave, Dashboard & Integration

### 3.1 `orchestrator/main.py` — Pipeline Entry Point

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

### 3.2 CopilotKit Race Dashboard

**`ui/components/RaceDashboard.tsx`** — shows live agent cards:

```typescript
// Each card shows:
// - Agent name (e.g. "Agent 2 — SAC — HalfCheetah")
// - Status badge: "starting" (gray), "training" (blue pulse), 
//                 "completed" (green), "failed/restarted" (red)
// - Progress bar: time elapsed / time budget
// - Current reward: latest heartbeat value, updates every 30s
// - Sentinel alert overlay if agent was nudged or restarted
```

Use `useCoAgentStateRender` to subscribe to state updates from the Orchestrator. The Orchestrator should push state after every heartbeat read cycle.

**`ui/components/SentinelAlert.tsx`:**

```typescript
// Renders as a warning card in the CopilotKit chat when Sentinel fires:
// "⚠ Agent 4 (GRPO lr=1.0) — NaN loss detected. Killing and restarting with lr=3e-4."
// Use useCopilotAction with name "sentinel_alert"
```

**`ui/components/ModelViewer.tsx`:**

After evaluation, render two sections:

1. **MuJoCo:** HTML5 `<video>` player loading `results/best_mujoco.mp4` from Person B's render script
2. **Countdown:** animated step-by-step solver — display each puzzle, then reveal the model's chain-of-thought and whether it succeeded. Use a simple card layout with the puzzle `[4, 7, 2, 9] → 24` and the model's reasoning steps below it

### 3.3 Checkpoint — Hour 10: Weave Trace Review

Run the pipeline with a mock training script that writes dummy `eval_result.json` after 10 seconds. Verify in Weave:

- `AutoRL_Pipeline` → `Orchestrator` → `SwarmRunner` → `TrainingAgent_agent_1`, `TrainingAgent_agent_2`, `DoomLoopSentinel` → `Evaluator` all appear as named nodes in the trace tree
- Online eval chart shows streaming values from the mock callback

If the trace looks flat (no hierarchy), check that `@weave.op` calls are nested inside parent `@weave.op` calls, not called independently.

---

## Phase 4 — Hours 14–28: Evaluator, Reporter & Polish

### 4.1 `evaluator/evaluator_agent.py`

```python
@weave.op(name="Evaluator")
async def evaluate_results(results: list[EvalResult]) -> dict:
    # Group by environment family
    mujoco_results = [r for r in results if r.env != "Countdown"]
    countdown_results = [r for r in results if r.env == "Countdown"]
    
    rankings = {}
    
    for group_name, group in [("MuJoCo", mujoco_results), ("Countdown", countdown_results)]:
        if not group:
            continue
        
        # LLM call to reason about rankings (NOT a fixed formula)
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
    
    # Push to Weave Evaluation
    _push_weave_evaluation(results, rankings)
    
    with open("rankings.json", "w") as f:
        json.dump(rankings, f)
    
    return rankings
```

### 4.2 Weave Evaluation Integration

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

### 4.3 `evaluator/reporter.py`

Template fill — no LLM call needed here. Just format `rankings.json` into `run_report.md`:

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
- Agent 4 (GRPO lr=1.0): NaN loss at step 45 → killed + restarted with lr=3e-4
```

### 4.4 Hour 16 Hard Gate

> **STOP AND VERIFY:** PPO + SAC on HalfCheetah, both local, through the complete pipeline: Orchestrator → spawn → Training Agents → heartbeats → Sentinel monitoring → Evaluator → rankings → CopilotKit UI showing results. If this is not green by Hour 16, do not touch RunPod until it is.

### 4.5 Model-in-Action Trigger

After Evaluator returns, the Orchestrator:

1. Identifies the best MuJoCo checkpoint path from `rankings.json`
2. Calls `subprocess.run(["python", "model_viewer/render_mujoco.py", "--checkpoint", path, "--output", "results/best_mujoco.mp4"])`
3. Identifies the best Countdown checkpoint path
4. Calls `subprocess.run(["python", "model_viewer/countdown_inference.py", "--checkpoint", path, "--output", "results/countdown_solve.json"])`
5. Sends both file paths to CopilotKit for the `ModelViewer` component to render

---

## Phase 5 — Hours 28–36: Demo Prep & Submission

### 5.1 Sentinel Demo Verification

Confirm that `agent_4` (with `lr=1.0`) triggers the Sentinel during every run. The heartbeat should show `anomaly: "nan_loss"` within 30-60 seconds of training start. Verify the Sentinel alert card appears in CopilotKit.

### 5.2 Set Weave Project to Public

In the W&B dashboard: Settings → Privacy → set to Public. Copy the Weave project URL into `run_report.md` and the submission form.

### 5.3 Demo Dry Run (Hour 30, with Person B)

- Run full pipeline from cold start
- Time it — should complete in under 15 minutes
- Verify all beats: approval card → race dashboard → Sentinel alert → leaderboard → model in action
- Record demo video immediately after a successful run

### 5.4 README.md

```markdown
# AutoRL — Multi-Agent RL Orchestration

AutoRL takes a natural language RL task and races competing algorithms in parallel,
supervised by a Doom Loop Sentinel that detects and recovers stuck agents.

## Architecture
[architecture diagram or screenshot]

## Setup
git clone ...
pip install -e .
cp .env.template .env  # fill in your keys
cd ui && npm install

## Run
python orchestrator/main.py
# or: cd ui && npm run dev

## Weave project: [URL]
## Demo video: [URL]
```

---

*Person A Build Guide — AutoRL Hackathon*
