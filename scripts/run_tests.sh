#!/bin/bash
# Module 13: Run Tests
# Unix/Linux/macOS bash script

echo "Running pytest tests..."

# Set PYTHONPATH to project root
export PYTHONPATH="$PWD"

# Run pytest with quiet mode
pytest -q
