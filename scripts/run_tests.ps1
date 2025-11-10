# PowerShell script to run pytest
# Windows-first approach for L3 baseline

$env:PYTHONPATH = $PWD
Write-Host "Running tests for Conversational RAG with Memory..."
Write-Host "PYTHONPATH set to: $env:PYTHONPATH"
Write-Host ""

pytest -q tests/
