#!/bin/bash
# Run API Server
# Usage: ./scripts/run_api.sh

export PYTHONPATH=$PWD
uvicorn app:app --reload --host 0.0.0.0 --port 8000
