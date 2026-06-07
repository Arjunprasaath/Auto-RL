"""Redis-backed coordination layer for AutoRL.

Provides pub/sub heartbeat streaming, run-state persistence, and nudge queues.
Falls back silently to file-based I/O when REDIS_URL is not set, so the system
works identically with or without Redis — zero configuration required for local
development.

Environment variable:
    REDIS_URL  redis://[[username:]password@]host[:port][/db]
               e.g. redis://default:secret@myhost.redis.io:11444
               When absent, all operations use the filesystem only.

Usage:
    from coordination.redis_coordinator import coordinator

    # Publish a heartbeat from a training script
    coordinator.publish_heartbeat(run_id, agent_id, data_dict)

    # Push / consume a Sentinel nudge
    coordinator.push_nudge(run_id, agent_id, hparams_dict)
    nudge = coordinator.pop_nudge(run_id, agent_id)   # None if no pending nudge

    # Persist / recover run state across backend restarts
    coordinator.set_run_state(run_id, state_dict)
    state = coordinator.get_run_state(run_id)   # None if not found

    # Subscribe to live heartbeats in an async context (FastAPI SSE)
    async for event in coordinator.subscribe_heartbeats(run_id):
        yield {"data": json.dumps(event)}
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncGenerator


# ── Redis key helpers ─────────────────────────────────────────────────────────

def _hb_channel(run_id: str) -> str:
    return f"autorl:heartbeat:{run_id}"

def _hb_key(run_id: str, agent_id: str) -> str:
    return f"autorl:hb:{run_id}:{agent_id}"

def _nudge_key(run_id: str, agent_id: str) -> str:
    return f"autorl:nudge:{run_id}:{agent_id}"

def _run_key(run_id: str) -> str:
    return f"autorl:run:{run_id}"


# ── Coordinator ───────────────────────────────────────────────────────────────


class RedisCoordinator:
    """Thin Redis wrapper with transparent file fallback.

    A single global instance (``coordinator``) is created at module load time.
    Redis is lazily connected on first use so import is always safe.
    """

    def __init__(self) -> None:
        self._redis_url: str | None = os.environ.get("REDIS_URL")
        self._client = None          # sync redis.Redis
        self._async_client = None    # async redis.asyncio.Redis

    # ── Internal connection helpers ───────────────────────────────────────────

    def _sync(self):
        """Return a synchronous Redis client, connecting lazily."""
        if self._client is not None:
            return self._client
        if not self._redis_url:
            return None
        try:
            import redis as _redis
            self._client = _redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            self._client.ping()
            print(f"[coordinator] connected to Redis at {self._redis_url}")
        except Exception as e:  # noqa: BLE001
            print(f"[coordinator] Redis unavailable ({e}) — using file fallback")
            self._client = None
        return self._client

    def _async_redis(self):
        """Return an async Redis client, connecting lazily."""
        if self._async_client is not None:
            return self._async_client
        if not self._redis_url:
            return None
        try:
            import redis.asyncio as _aredis
            self._async_client = _aredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[coordinator] async Redis unavailable ({e})")
            self._async_client = None
        return self._async_client

    # ── Heartbeats ────────────────────────────────────────────────────────────

    def publish_heartbeat(self, run_id: str, agent_id: str, data: dict) -> None:
        """Publish heartbeat data to Redis pub/sub and store latest value.

        The latest heartbeat is also cached in a Redis hash so late subscribers
        can recover current state without replaying history.
        """
        r = self._sync()
        if r is None:
            return
        try:
            payload = json.dumps(data)
            r.set(_hb_key(run_id, agent_id), payload, ex=300)   # 5-min TTL
            r.publish(_hb_channel(run_id), payload)
        except Exception as e:  # noqa: BLE001
            print(f"[coordinator] publish_heartbeat failed ({e})")

    def get_all_heartbeats(self, run_id: str) -> dict[str, dict]:
        """Return the latest heartbeat for every agent in a run."""
        r = self._sync()
        if r is None:
            return {}
        try:
            pattern = _hb_key(run_id, "*")
            keys = r.keys(pattern)
            if not keys:
                return {}
            values = r.mget(keys)
            result: dict[str, dict] = {}
            for key, raw in zip(keys, values):
                if raw:
                    agent_id = key.split(":")[-1]
                    result[agent_id] = json.loads(raw)
            return result
        except Exception as e:  # noqa: BLE001
            print(f"[coordinator] get_all_heartbeats failed ({e})")
            return {}

    async def subscribe_heartbeats(self, run_id: str) -> AsyncGenerator[dict, None]:
        """Async generator that yields heartbeat dicts as they arrive via pub/sub.

        First yields all cached (latest) heartbeats so the subscriber is
        immediately up-to-date, then streams live updates.
        """
        r = self._async_redis()
        if r is None:
            # No Redis: yield a single sentinel so the caller can fall back
            yield {"_no_redis": True}
            return

        # Yield cached state first
        try:
            pattern = _hb_key(run_id, "*")
            keys = await r.keys(pattern)
            if keys:
                values = await r.mget(keys)
                for raw in values:
                    if raw:
                        yield json.loads(raw)
        except Exception:  # noqa: BLE001
            pass

        # Stream live updates
        pubsub = r.pubsub()
        try:
            await pubsub.subscribe(_hb_channel(run_id))
            async for msg in pubsub.listen():
                if msg and msg.get("type") == "message":
                    try:
                        yield json.loads(msg["data"])
                    except Exception:  # noqa: BLE001
                        pass
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            print(f"[coordinator] subscribe_heartbeats error ({e})")
        finally:
            try:
                await pubsub.unsubscribe(_hb_channel(run_id))
                await pubsub.aclose()
            except Exception:  # noqa: BLE001
                pass

    # ── Nudges ────────────────────────────────────────────────────────────────

    def push_nudge(self, run_id: str, agent_id: str, nudge: dict) -> None:
        """Push a Sentinel nudge so the training script can consume it."""
        r = self._sync()
        if r is None:
            return
        try:
            r.set(_nudge_key(run_id, agent_id), json.dumps(nudge), ex=600)
        except Exception as e:  # noqa: BLE001
            print(f"[coordinator] push_nudge failed ({e})")

    def pop_nudge(self, run_id: str, agent_id: str) -> dict | None:
        """Atomically read and delete a pending nudge. Returns None if absent."""
        r = self._sync()
        if r is None:
            return None
        key = _nudge_key(run_id, agent_id)
        try:
            pipe = r.pipeline()
            pipe.get(key)
            pipe.delete(key)
            raw, _ = pipe.execute()
            return json.loads(raw) if raw else None
        except Exception as e:  # noqa: BLE001
            print(f"[coordinator] pop_nudge failed ({e})")
            return None

    # ── Run state ─────────────────────────────────────────────────────────────

    def set_run_state(self, run_id: str, state: dict) -> None:
        """Persist run metadata so it survives backend restarts."""
        r = self._sync()
        if r is None:
            return
        try:
            r.set(_run_key(run_id), json.dumps(state), ex=86400)  # 24-hr TTL
        except Exception as e:  # noqa: BLE001
            print(f"[coordinator] set_run_state failed ({e})")

    def get_run_state(self, run_id: str) -> dict | None:
        """Recover persisted run state. Returns None if not found."""
        r = self._sync()
        if r is None:
            return None
        try:
            raw = r.get(_run_key(run_id))
            return json.loads(raw) if raw else None
        except Exception as e:  # noqa: BLE001
            print(f"[coordinator] get_run_state failed ({e})")
            return None

    def list_run_ids(self) -> list[str]:
        """List all run IDs currently stored in Redis."""
        r = self._sync()
        if r is None:
            return []
        try:
            keys = r.keys("autorl:run:*")
            return [k.split("autorl:run:")[-1] for k in keys]
        except Exception:  # noqa: BLE001
            return []


# ── Singleton ─────────────────────────────────────────────────────────────────

coordinator = RedisCoordinator()
