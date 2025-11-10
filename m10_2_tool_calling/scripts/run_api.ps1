# PowerShell script to run FastAPI server
# Usage: .\scripts\run_api.ps1

Write-Host "Starting FastAPI server for Module 10.2..." -ForegroundColor Green

# Set Python path to include current directory
$env:PYTHONPATH = $PWD

# Run uvicorn with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# If uvicorn not found, provide helpful message
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nError: uvicorn not found. Install dependencies:" -ForegroundColor Red
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
}
