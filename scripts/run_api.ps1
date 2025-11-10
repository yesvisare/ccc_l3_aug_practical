# PowerShell script to run the Usage Metering API
# Sets PYTHONPATH to current directory and starts uvicorn with hot-reload

Write-Host "Starting Usage Metering & Analytics API..." -ForegroundColor Green

# Set PYTHONPATH to current directory
$env:PYTHONPATH = $PWD

# Run uvicorn with reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# If uvicorn is not found, provide helpful message
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Error: uvicorn not found!" -ForegroundColor Red
    Write-Host "Install dependencies with: pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}
