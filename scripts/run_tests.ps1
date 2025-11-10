# PowerShell script to run pytest tests
# Usage: .\scripts\run_tests.ps1

Write-Host "Running pytest tests..." -ForegroundColor Green

# Set PYTHONPATH to project root
$env:PYTHONPATH = $PWD

# Run pytest with quiet mode
pytest -q tests/

Write-Host "Tests completed." -ForegroundColor Yellow
