# Run pytest suite
# Usage: .\scripts\run_tests.ps1

$env:PYTHONPATH = $PWD
pytest -v tests/
