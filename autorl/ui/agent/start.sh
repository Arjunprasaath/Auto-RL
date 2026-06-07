#!/usr/bin/env bash
# Start the AutoRL Python FastAPI backend.
# Run from the autorl/ directory:
#   bash ui/agent/start.sh
set -e
cd "$(dirname "$0")/../.."   # go to autorl/
source .venv/bin/activate
uvicorn ui.agent.middleware:app --host 0.0.0.0 --port 8000 --reload
