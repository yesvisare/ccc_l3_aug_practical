# Run pytest tests
# Usage: .\scripts\run_tests.ps1

Write-Host "Running tests..." -ForegroundColor Green

# Set PYTHONPATH to include current directory
$env:PYTHONPATH = $PWD

# Run pytest in quiet mode
pytest -q

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nAll tests passed!" -ForegroundColor Green
} else {
    Write-Host "`nSome tests failed. Check output above." -ForegroundColor Red
}
