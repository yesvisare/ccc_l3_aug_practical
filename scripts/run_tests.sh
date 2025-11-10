#!/bin/bash
# Multi-Agent Orchestration Test Runner

echo "Running Multi-Agent Orchestration Tests..."
export PYTHONPATH="$PWD"
python3 -m pytest tests/ -q
