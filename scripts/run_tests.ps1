# Run pytest tests
# Usage: .\scripts\run_tests.ps1

$env:PYTHONPATH = $PWD
pytest -q
