# PowerShell script to run tests
# Usage: .\scripts\run_tests.ps1

Write-Host "Running tests for Module 10.2..." -ForegroundColor Green

# Set Python path to include current directory
$env:PYTHONPATH = $PWD

# Run pytest with quiet output
pytest -q

# Show summary
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ All tests passed!" -ForegroundColor Green
} else {
    Write-Host "`n⚠️  Some tests failed. See output above." -ForegroundColor Yellow
}

# If pytest not found, provide helpful message
if ($LASTEXITCODE -eq 127) {
    Write-Host "`nError: pytest not found. Install dependencies:" -ForegroundColor Red
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
}
