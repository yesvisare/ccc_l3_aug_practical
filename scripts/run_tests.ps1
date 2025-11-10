# Module 13.2: Governance & Compliance Documentation
# PowerShell script to run pytest tests

Write-Host "Running compliance documentation tests..." -ForegroundColor Cyan

# Set PYTHONPATH to current directory for imports
$env:PYTHONPATH = $PWD

# Run pytest with quiet mode
pytest -q

# Alternative options:
# pytest -v              # Verbose output
# pytest --cov           # With coverage report
# pytest -k test_name    # Run specific test
