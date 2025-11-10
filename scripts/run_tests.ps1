# PowerShell script to run tests
# Sets PYTHONPATH to current directory and runs pytest

Write-Host "Running Module 12.1 Tests..." -ForegroundColor Green

# Set PYTHONPATH to current directory
$env:PYTHONPATH = $PWD

# Run pytest with quiet mode
pytest -q

# Show result
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ All tests passed!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "✗ Some tests failed!" -ForegroundColor Red
    exit 1
}
