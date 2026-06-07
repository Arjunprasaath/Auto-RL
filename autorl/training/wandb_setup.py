"""
Weights & Biases logging for SB3 training.

Enables per-step metric charts (reward, loss, entropy, etc.) in W&B by syncing
Stable-Baselines3's TensorBoard output and attaching `WandbCallback`. This is
separate from Weave tracing (which captures the @weave.op call tree) — together
they give both the trace and the time-series graphs.

Gated on WANDB_API_KEY; disabled when WANDB_DISABLED=1 (independent of Weave, so
turning off tracing never kills your charts). Safe no-op if wandb isn't installed.
"""

import os


def wandb_enabled() -> bool:
    if os.environ.get("WANDB_DISABLED"):
        return False
    return bool(os.environ.get("WANDB_API_KEY"))


def start_wandb_run(agent_id, algo, env_id, lr, seed, results_dir):
    """Start a W&B run for this agent.

    Returns (run, tensorboard_log_path, sb3_callback). All three are None when
    logging is disabled or wandb is unavailable, so callers can branch simply.
    """
    if not wandb_enabled():
        return None, None, None
    try:
        import wandb
        from wandb.integration.sb3 import WandbCallback
    except ImportError:
        print("[wandb] not installed — per-step charts disabled")
        return None, None, None

    project = os.environ.get("WEAVE_PROJECT", "autorl")
    tb_log = os.path.join(results_dir, agent_id, "tb")
    try:
        run = wandb.init(
            project=project,
            name=agent_id,
            group=env_id,
            config={"algo": algo, "env": env_id, "lr": lr, "seed": seed},
            sync_tensorboard=True,  # mirror SB3 TensorBoard scalars as W&B charts
            reinit="finish_previous",
        )
    except Exception as e:  # noqa: BLE001 - local resilience
        print(f"[wandb] init skipped ({e})")
        return None, None, None

    return run, tb_log, WandbCallback(verbose=0)


def find_warm_start_checkpoint(algo: str, env_id: str, results_dir: str) -> str | None:
    """Download the best past model artifact from W&B to warm-start training.

    Queries the W&B API for the highest-scoring finished run matching this
    algo+env, downloads its model artifact to a local cache, and returns the
    path to the .zip file.  Returns None when W&B is disabled, no matching
    past run exists, or the download fails — callers always fall back to
    training from scratch.
    """
    if not wandb_enabled():
        return None
    entity = os.environ.get("WANDB_ENTITY", "")
    project = os.environ.get("WEAVE_PROJECT", "autorl")
    if not entity:
        return None
    try:
        import wandb
        api = wandb.Api(timeout=15)
        runs = api.runs(
            f"{entity}/{project}",
            filters={
                "config.algo": algo,
                "config.env": env_id,
                "state": "finished",
            },
            order="-summary_metrics.mean_return",
            per_page=1,
        )
        best_run = next(iter(runs), None)
        if best_run is None:
            print(f"[wandb] warm-start: no finished runs for {algo}/{env_id}")
            return None
        artifacts = [a for a in best_run.logged_artifacts() if a.type == "model"]
        if not artifacts:
            print(f"[wandb] warm-start: no model artifact in run {best_run.id}")
            return None
        art = artifacts[0]
        warm_dir = os.path.join(results_dir, "_warm_start_cache", f"{algo}_{env_id}")
        os.makedirs(warm_dir, exist_ok=True)
        art.download(root=warm_dir)
        for fname in os.listdir(warm_dir):
            if fname.endswith(".zip"):
                ckpt_path = os.path.join(warm_dir, fname)
                ret = best_run.summary.get("mean_return")
                ret_str = f"{ret:.1f}" if ret is not None else "?"
                print(f"[wandb] warm-start: {ckpt_path} (run {best_run.id}, return={ret_str})")
                return ckpt_path
        print(f"[wandb] warm-start: download succeeded but no .zip in {warm_dir}")
        return None
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] warm-start lookup failed ({e})")
        return None


def log_model_artifact(
    run,
    checkpoint_path: str,
    agent_id: str,
    metadata: dict | None = None,
) -> None:
    """Log a model checkpoint as a W&B Artifact for versioning and comparison.

    Safe no-op when run is None (W&B disabled) or the file doesn't exist yet.
    """
    if run is None:
        return
    import os as _os
    if not _os.path.exists(checkpoint_path):
        print(f"[wandb] artifact skipped — checkpoint not found: {checkpoint_path}")
        return
    try:
        import wandb
        art = wandb.Artifact(
            name=agent_id,
            type="model",
            metadata=metadata or {},
        )
        art.add_file(checkpoint_path)
        run.log_artifact(art)
        print(f"[wandb] artifact logged: {agent_id} → {checkpoint_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] artifact logging failed ({e})")


def finish_wandb_run(
    run,
    mean_return: float | None = None,
    std_return: float | None = None,
    checkpoint: str | None = None,
) -> None:
    """Finish the W&B run and write final metrics to the run summary."""
    if run is not None:
        try:
            summary: dict = {}
            if mean_return is not None:
                summary["mean_return"] = mean_return
            if std_return is not None:
                summary["std_return"] = std_return
            if checkpoint:
                summary["checkpoint"] = checkpoint
            if summary:
                run.summary.update(summary)
            run.finish()
        except Exception:  # noqa: BLE001
            pass
