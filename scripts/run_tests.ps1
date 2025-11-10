# Run pytest tests
# Sets PYTHONPATH to current directory and runs pytest in quiet mode

$env:PYTHONPATH = $PWD
pytest -q tests/
