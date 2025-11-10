#!/bin/bash
# Run FastAPI application
# Sets PYTHONPATH to current directory and starts uvicorn with reload

export PYTHONPATH=$PWD
uvicorn app:app --reload --host 0.0.0.0 --port 8000
