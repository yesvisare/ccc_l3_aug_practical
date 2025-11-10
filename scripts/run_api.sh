#!/bin/bash
# Run the FastAPI application with auto-reload
# Usage: ./scripts/run_api.sh

echo "Starting FastAPI application..."

# Set PYTHONPATH to include current directory
export PYTHONPATH=$PWD

# Run uvicorn with reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

if [ $? -ne 0 ]; then
    echo "Failed to start API server. Check that uvicorn is installed."
    echo "Install with: pip install uvicorn[standard]"
fi
