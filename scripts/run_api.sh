#!/bin/bash
# Module 13: Run FastAPI Server
# Unix/Linux/macOS bash script

echo "Starting FastAPI server..."

# Set PYTHONPATH to project root
export PYTHONPATH="$PWD"

# Run uvicorn with reload for development
uvicorn app:app --reload --host 0.0.0.0 --port 8000
