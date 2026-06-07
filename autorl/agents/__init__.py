"""AutoRL agents package.

Owns:
  agents.sentinel       — LLM-based Doom Loop Sentinel
  agents.training_agent — asyncio subprocess wrapper for training scripts

Also re-exports the openai-agents SDK (Agent, AgentOutputSchema, Runner, …)
so that ``from agents import Agent`` continues to work despite this directory
shadowing the installed ``agents`` package name.

Bootstrap strategy
──────────────────
When Python first imports this package, sys.modules["agents"] is set to this
module object (partially initialised).  We locate the SDK's __init__.py in
site-packages, build a module whose __path__ points at the SDK directory, then
temporarily swap it into sys.modules["agents"] so the SDK's own
``from agents.xxx import …`` submodule lookups resolve inside its own
directory.  After exec we copy all public symbols into our namespace, then
restore ourselves as sys.modules["agents"] so agents.sentinel etc. still work.
"""

from __future__ import annotations

import importlib.util
import os
import sys


def _bootstrap_sdk() -> None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    this_module = sys.modules.get("agents")  # ourselves, partially initialised

    for search_path in sys.path:
        sdk_init = os.path.join(search_path, "agents", "__init__.py")
        sdk_dir = os.path.join(search_path, "agents")
        if not os.path.isfile(sdk_init):
            continue
        if os.path.abspath(sdk_dir) == this_dir:
            continue  # that's us — skip

        spec = importlib.util.spec_from_file_location(
            "agents",
            sdk_init,
            submodule_search_locations=[sdk_dir],
        )
        if spec is None or spec.loader is None:
            continue

        sdk_mod = importlib.util.module_from_spec(spec)
        sdk_mod.__path__ = [sdk_dir]  # type: ignore[assignment]

        # Register SDK as "agents" so its own relative imports resolve correctly.
        sys.modules["agents"] = sdk_mod
        try:
            spec.loader.exec_module(sdk_mod)  # type: ignore[union-attr]
        except Exception:
            sys.modules["agents"] = this_module  # rollback on failure
            raise

        # Copy every public symbol the SDK exposes into our namespace.
        for name in dir(sdk_mod):
            if not name.startswith("_"):
                globals()[name] = getattr(sdk_mod, name)

        # Restore ourselves as "agents" so agents.sentinel / agents.training_agent work.
        if this_module is not None:
            sys.modules["agents"] = this_module
        return


_bootstrap_sdk()
