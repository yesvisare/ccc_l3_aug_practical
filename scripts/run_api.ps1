# Module 13.2: Governance & Compliance Documentation
# PowerShell script to run the FastAPI application

Write-Host "Starting Compliance Documentation API..." -ForegroundColor Cyan

# Set PYTHONPATH to current directory
$env:PYTHONPATH = $PWD

# Run uvicorn with reload for development
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Alternative for production (no reload):
# uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
