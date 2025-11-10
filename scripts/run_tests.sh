#!/bin/bash
# Run pytest tests
# Usage: ./scripts/run_tests.sh

echo "Running tests..."

# Set PYTHONPATH to include current directory
export PYTHONPATH=$PWD

# Run pytest in quiet mode
pytest -q

if [ $? -eq 0 ]; then
    echo ""
    echo "All tests passed!"
else
    echo ""
    echo "Some tests failed. Check output above."
fi
