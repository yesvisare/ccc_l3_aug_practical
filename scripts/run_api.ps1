# Run the FastAPI application with auto-reload
# Usage: .\scripts\run_api.ps1

Write-Host "Starting FastAPI application..." -ForegroundColor Green

# Set PYTHONPATH to include current directory
$env:PYTHONPATH = $PWD

# Run uvicorn with reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start API server. Check that uvicorn is installed." -ForegroundColor Red
    Write-Host "Install with: pip install uvicorn[standard]" -ForegroundColor Yellow
}
