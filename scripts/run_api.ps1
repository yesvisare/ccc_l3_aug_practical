# Run the FastAPI application for Module 12.4
# Tenant Lifecycle Management API

Write-Host "Starting Tenant Lifecycle Management API..." -ForegroundColor Green

# Set PYTHONPATH to include project root
$env:PYTHONPATH = $PWD

# Run uvicorn with reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Usage:
# .\scripts\run_api.ps1
# Then open http://localhost:8000/docs for API documentation
