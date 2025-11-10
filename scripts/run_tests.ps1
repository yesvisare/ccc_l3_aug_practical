# Module 13.3: Launch Preparation & Marketing
# PowerShell script to run pytest tests

Write-Host "Running Module 13.3 Tests..." -ForegroundColor Green

# Set PYTHONPATH to include project root
$env:PYTHONPATH = $PWD

# Run pytest with quiet mode
pytest -q tests/

# For verbose output, use:
# pytest -v tests/

# For coverage report:
# pytest --cov=src tests/

# Usage:
# PS> .\scripts\run_tests.ps1
