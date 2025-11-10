# Module 7.2: Run FastAPI Server
# Windows PowerShell script

Write-Host "Starting Module 7.2 API Server..." -ForegroundColor Green

# Set PYTHONPATH to project root
$env:PYTHONPATH = $PWD

# Run uvicorn with reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# If uvicorn fails
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to start API server" -ForegroundColor Red
    Write-Host "Make sure uvicorn is installed: pip install uvicorn[standard]" -ForegroundColor Yellow
    exit 1
}
