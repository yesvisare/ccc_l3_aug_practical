# Module 13.3: Launch Preparation & Marketing
# PowerShell script to run the FastAPI application

Write-Host "Starting Module 13.3 API Server..." -ForegroundColor Green

# Set PYTHONPATH to include project root
$env:PYTHONPATH = $PWD

# Run FastAPI with uvicorn (reload enabled for development)
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Usage:
# PS> .\scripts\run_api.ps1
# Then visit: http://localhost:8000/docs
