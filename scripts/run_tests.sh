#!/bin/bash
# Run Tests
# Usage: ./scripts/run_tests.sh

export PYTHONPATH=$PWD
pytest -v tests/
