"""
HeartbeatWriter — background thread that writes heartbeat.json every 60 seconds.

Imported by ALL training scripts (train_ppo.py, train_sac.py, train_a2c.py,
train_grpo_countdown.py). This is the file the Doom Loop Sentinel monitors.

Usage:
    hb = HeartbeatWriter(agent_id="agent_1", results_dir="./results")
    hb.start()
    # ... inside training loop:
    hb.update(steps=total_steps, reward=last_reward, loss=current_loss)
    nudge = hb.check_nudge()
    if nudge:
        apply_new_lr(nudge["lr"])
    # ... at the end:
    hb.stop("completed")
"""

import json
import threading
import time
import os
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
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        """Start the background heartbeat thread. Call before training begins."""
        with self._lock:
            self.status = "training"
        self._thread.start()

    def update(self, steps: int, reward: float, loss: float = None):
        """
        Update the current training metrics.
        Call this from the training loop after each chunk of steps.
        NaN/inf loss is automatically detected and sets anomaly="nan_loss".
        """
        with self._lock:
            self.steps = steps
            self.reward = reward
            self.loss = loss
            if loss is not None:
                # Detect NaN (loss != loss) or explosion (|loss| > 1e6)
                if loss != loss or abs(loss) > 1e6:
                    self.anomaly = "nan_loss"

    def stop(self, status: str = "completed"):
        """
        Write a final heartbeat with the given status and stop the thread.
        Call at the end of training (success or failure).
        """
        with self._lock:
            self.status = status
        self._write()
        self._stop.set()
        self._thread.join(timeout=5)

    def check_nudge(self) -> dict | None:
        """
        Check if the Sentinel has written a nudge.json with new hyperparameters.
        If found, reads it, deletes it, and returns the hparams dict.
        Returns None if no nudge is pending.

        Call this in the training loop on every iteration:
            nudge = hb.check_nudge()
            if nudge:
                new_lr = nudge.get("lr", current_lr)
                optimizer.param_groups[0]["lr"] = new_lr
        """
        if self.nudge_path.exists():
            try:
                with open(self.nudge_path) as f:
                    nudge = json.load(f)
                self.nudge_path.unlink()
                print(f"[{self.agent_id}] Nudge received: {nudge}")
                return nudge
            except Exception as e:
                print(f"[{self.agent_id}] Failed to read nudge: {e}")
        return None

    def _loop(self):
        """Background loop: write heartbeat every 60 seconds."""
        while not self._stop.is_set():
            self._write()
            self._stop.wait(60)

    def _write(self):
        """Write the current state to heartbeat.json atomically."""
        with self._lock:
            data = {
                "agent_id": self.agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": self.status,
                "steps_completed": self.steps,
                "current_reward": self.reward,
                "loss": self.loss,
                "anomaly": self.anomaly,
            }

        # Write to a temp file first, then rename — avoids partial reads by Sentinel
        tmp_path = self.hb_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.hb_path)
