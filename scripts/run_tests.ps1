# Module 13: Run Tests
# Windows PowerShell script

Write-Host "Running pytest tests..." -ForegroundColor Green

# Set PYTHONPATH to project root
$env:PYTHONPATH = $PWD

# Run pytest with quiet mode
pytest -q
