# Run pytest with quiet mode
# Usage: .\scripts\run_tests.ps1

$env:PYTHONPATH = $PWD
pytest -q tests/
