#!/bin/bash
# Run the FastAPI application for Module 12.4
# Tenant Lifecycle Management API

echo "Starting Tenant Lifecycle Management API..."

# Set PYTHONPATH to include project root
export PYTHONPATH=$(pwd)

# Run uvicorn with reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Usage:
# ./scripts/run_api.sh
# Then open http://localhost:8000/docs for API documentation
