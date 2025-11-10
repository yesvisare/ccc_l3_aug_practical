# Module 13: Run FastAPI Server
# Windows PowerShell script

Write-Host "Starting FastAPI server..." -ForegroundColor Green

# Set PYTHONPATH to project root
$env:PYTHONPATH = $PWD

# Run uvicorn with reload for development
uvicorn app:app --reload --host 0.0.0.0 --port 8000
