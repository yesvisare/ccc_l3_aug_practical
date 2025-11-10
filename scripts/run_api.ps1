# PowerShell script to run the FastAPI application
# Usage: .\scripts\run_api.ps1

Write-Host "Starting FastAPI application..." -ForegroundColor Green

# Set PYTHONPATH to project root
$env:PYTHONPATH = $PWD

# Run uvicorn with reload for development
uvicorn app:app --reload --host 0.0.0.0 --port 8000

Write-Host "API server stopped." -ForegroundColor Yellow
