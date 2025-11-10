#!/bin/bash
# Multi-Agent Orchestration API Server

echo "Starting Multi-Agent Orchestration API..."
export PYTHONPATH="$PWD"
uvicorn app:app --reload --host 0.0.0.0 --port 8000
