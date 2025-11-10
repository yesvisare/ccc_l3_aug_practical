# Run pytest tests
# Usage: .\scripts\run_tests.ps1

$env:PYTHONPATH = $PWD
Write-Host "Running pytest with PYTHONPATH: $env:PYTHONPATH"
Write-Host ""

pytest -q tests/
