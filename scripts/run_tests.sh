#!/bin/bash
# Run pytest tests
# Sets PYTHONPATH to current directory and runs pytest in quiet mode

export PYTHONPATH=$PWD
pytest -q tests/
