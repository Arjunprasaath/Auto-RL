"""
Pydantic schemas for AutoRL data contracts.

This is the interface between Person A (orchestration) and Person B (training).
- Person A writes spawn_plan.json using SpawnPlanEntry
- Person B writes heartbeat.json using Heartbeat
- Person B writes eval_result.json using EvalResult
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime


class SpawnPlanEntry(BaseModel):
    """Defines a single training agent to spawn."""
    
    id: str  # e.g., "agent_1"
    algo: str  # "PPO", "SAC", "A2C", "GRPO"
    env: str  # "HalfCheetah-v5", "Hopper-v5", "Countdown"
    exec: Literal["local", "runpod"]
    time_budget_min: int  # 10 for MuJoCo, 20 for Countdown
    hparams: dict = Field(default_factory=dict)


class Heartbeat(BaseModel):
    """Training agent status update, written every 60s."""
    
    agent_id: str
    timestamp: datetime
    status: Literal["starting", "training", "completed", "failed", "restarted"]
    steps_completed: int = 0
    current_reward: float = 0.0
    loss: Optional[float] = None
    anomaly: Optional[str] = None  # "nan_loss", "plateau", None


class EvalResult(BaseModel):
    """Final training result from a completed agent."""
    
    agent_id: str
    algo: str
    env: str
    status: Literal["completed", "failed", "timed_out", "restarted"]
    mean_return: float = 0.0
    std_return: float = 0.0
    steps_trained: int = 0
    wall_time_s: float = 0.0
    weave_run_id: str = ""
    checkpoint_path: str = ""


class NudgeConfig(BaseModel):
    """Configuration sent to a stuck agent via nudge.json."""
    
    lr: float
    seed: int
    message: str = ""
