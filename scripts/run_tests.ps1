# PowerShell script to run tests
# Sets PYTHONPATH and runs pytest

$env:PYTHONPATH = $PWD
Write-Host "Running tests..."
Write-Host "PYTHONPATH: $env:PYTHONPATH"
pytest -q tests/
