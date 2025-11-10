# Module 7.2: Run Tests
# Windows PowerShell script

Write-Host "Running Module 7.2 Tests..." -ForegroundColor Green

# Set PYTHONPATH to project root
$env:PYTHONPATH = $PWD

# Run pytest quietly
pytest tests/ -q

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nAll tests passed!" -ForegroundColor Green
} else {
    Write-Host "`nSome tests failed." -ForegroundColor Red
    Write-Host "Run 'pytest tests/ -v' for detailed output" -ForegroundColor Yellow
    exit 1
}
