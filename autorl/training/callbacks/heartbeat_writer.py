"""
HeartbeatWriter — background thread that writes heartbeat.json every 5 seconds.

Anomalies detected (in priority order, highest first):
  nan_loss                — loss is NaN or |loss| > 1e6 → kill + LLM restart
  critic_diverged         — explained_variance < -0.5 after 10k steps → kill + LLM restart
  plateau                 — reward stuck (<5% range) over last 10 chunks after 30k steps → nudge
  entropy_collapsed       — PPO entropy_loss > -0.5 before 50k steps → nudge
  episode_length_regression — ep length drops >40% after rising above 50 → nudge

All detected anomalies are included in heartbeat.json under "anomaly". The Sentinel
reads this field on every 30s check cycle and takes action.

Usage:
    hb = HeartbeatWriter(agent_id="agent_1", results_dir="./results")
    hb.start()
    # each chunk:
    hb.update(steps, reward, loss=loss,
              explained_variance=ev, entropy_loss=ent, ep_lengths=cb.ep_lengths)
    nudge = hb.check_nudge()
    # at end:
    hb.stop("completed")
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


# Anomaly priority — lower index = higher priority.
# Once a higher-priority anomaly is set it won't be overwritten by a lower one.
_PRIORITY = [
    "nan_loss",
    "critic_diverged",
    "plateau",
    "entropy_collapsed",
    "episode_length_regression",
]


def _higher_priority(new: str, current: str | None) -> bool:
    """Return True if new anomaly should replace current one."""
    if current is None:
        return True
    try:
        return _PRIORITY.index(new) < _PRIORITY.index(current)
    except ValueError:
        return False


class HeartbeatWriter:
    # Thresholds
    PLATEAU_WINDOW      = 10    # reward chunks to check
    PLATEAU_THRESHOLD   = 0.05  # max relative range to call it a plateau
    PLATEAU_MIN_STEPS   = 30_000
    EV_MIN_STEPS        = 10_000
    EV_THRESHOLD        = -0.5
    ENTROPY_MAX_STEPS   = 50_000
    ENTROPY_THRESHOLD   = -0.5  # train/entropy_loss > this → collapsed (near 0)
    EP_LEN_WINDOW       = 20    # episodes for regression check
    EP_LEN_DROP         = 0.6   # second half / first half ratio threshold

    def __init__(self, agent_id: str, results_dir: str = "./results"):
        self.agent_id = agent_id
        self.dir = Path(results_dir) / agent_id
        self.dir.mkdir(parents=True, exist_ok=True)

        self.hb_path    = self.dir / "heartbeat.json"
        self.nudge_path = self.dir / "nudge.json"

        self.steps    = 0
        self.reward   = 0.0
        self.loss     = None
        self.anomaly  = None
        self.status   = "starting"

        # Extra metrics included in heartbeat JSON (best-effort)
        self._explained_variance: float | None = None
        self._entropy_loss: float | None       = None
        self._extra: dict                      = {}  # arbitrary extra fields

        # Internal history for anomaly detection
        self._reward_history: list[float] = []

        self._stop   = threading.Event()
        self._lock   = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        with self._lock:
            self.status = "training"
        self._thread.start()

    def update(
        self,
        steps: int,
        reward: float,
        loss: float | None = None,
        *,
        explained_variance: float | None = None,
        entropy_loss: float | None = None,
        ep_lengths: list[float] | None = None,
    ) -> None:
        """Update training metrics and run anomaly detection.

        Args:
            steps:              total env steps so far
            reward:             latest episode return
            loss:               policy/actor loss (NaN triggers kill)
            explained_variance: train/explained_variance from SB3 logger (PPO/A2C)
            entropy_loss:       train/entropy_loss from SB3 logger (PPO/A2C)
            ep_lengths:         full episode-length history list from WeaveLogCallback
        """
        with self._lock:
            self.steps  = steps
            self.reward = reward
            self.loss   = loss
            if explained_variance is not None:
                self._explained_variance = explained_variance
            if entropy_loss is not None:
                self._entropy_loss = entropy_loss

            # Reward history (one entry per chunk)
            self._reward_history.append(reward)
            if len(self._reward_history) > self.PLATEAU_WINDOW:
                self._reward_history.pop(0)

            self._detect_anomalies(steps, loss, explained_variance, entropy_loss, ep_lengths)

    def _detect_anomalies(
        self,
        steps: int,
        loss: float | None,
        ev: float | None,
        entropy: float | None,
        ep_lengths: list[float] | None,
    ) -> None:
        """Run all anomaly checks in priority order. Called inside lock."""

        def _set(anomaly: str) -> None:
            if _higher_priority(anomaly, self.anomaly):
                self.anomaly = anomaly

        # 1. NaN / exploding loss
        if loss is not None and (loss != loss or abs(loss) > 1e6):
            _set("nan_loss")
            return  # highest priority — no need to check further

        # 2. Critic diverged (PPO/A2C: explained_variance persistently negative)
        if ev is not None and steps > self.EV_MIN_STEPS and ev < self.EV_THRESHOLD:
            _set("critic_diverged")

        # 3. Reward plateau (skip SAC warmup by requiring min steps)
        if steps > self.PLATEAU_MIN_STEPS and len(self._reward_history) == self.PLATEAU_WINDOW:
            rng      = max(self._reward_history) - min(self._reward_history)
            baseline = max(abs(self._reward_history[0]), 1.0)
            if rng / baseline < self.PLATEAU_THRESHOLD:
                _set("plateau")

        # 4. Entropy collapsed early (PPO/A2C; entropy_loss close to 0 means no exploration)
        if entropy is not None and steps < self.ENTROPY_MAX_STEPS and entropy > self.ENTROPY_THRESHOLD:
            _set("entropy_collapsed")

        # 5. Episode length regression (agent learned to survive, then forgot)
        if ep_lengths and len(ep_lengths) >= self.EP_LEN_WINDOW:
            first_half  = sum(ep_lengths[-self.EP_LEN_WINDOW: -self.EP_LEN_WINDOW // 2]) / (self.EP_LEN_WINDOW // 2)
            second_half = sum(ep_lengths[-self.EP_LEN_WINDOW // 2:]) / (self.EP_LEN_WINDOW // 2)
            if first_half > 50 and second_half < first_half * self.EP_LEN_DROP:
                _set("episode_length_regression")

    def set_extra(self, **kwargs) -> None:
        """Store arbitrary extra fields to include in every heartbeat write."""
        with self._lock:
            self._extra.update(kwargs)

    def stop(self, status: str = "completed") -> None:
        with self._lock:
            self.status = status
        self._write()
        self._stop.set()
        self._thread.join(timeout=5)

    def check_nudge(self) -> dict | None:
        """Return LLM-suggested hparams from nudge.json, deleting it after read."""
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

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._write()
            self._stop.wait(5)

    def _write(self) -> None:
        with self._lock:
            data = {
                "agent_id":           self.agent_id,
                "timestamp":          datetime.now(timezone.utc).isoformat(),
                "status":             self.status,
                "steps_completed":    self.steps,
                "current_reward":     self.reward,
                "loss":               self.loss,
                "anomaly":            self.anomaly,
                "explained_variance": self._explained_variance,
                "entropy_loss":       self._entropy_loss,
                **self._extra,
            }

        tmp = self.hb_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.hb_path)
