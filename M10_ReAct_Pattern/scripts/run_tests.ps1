# Run pytest for ReAct Pattern Implementation
# Sets PYTHONPATH to current directory and runs pytest in quiet mode

$env:PYTHONPATH = $PWD
Write-Host "Running tests for ReAct Pattern Implementation..."
Write-Host "PYTHONPATH set to: $env:PYTHONPATH"
Write-Host ""

pytest -q tests/
