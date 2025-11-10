# Run the FastAPI application
# Usage: .\scripts\run_api.ps1

$env:PYTHONPATH = $PWD
Write-Host "Starting FastAPI server on http://0.0.0.0:8000"
Write-Host "PYTHONPATH set to: $env:PYTHONPATH"
Write-Host ""

uvicorn app:app --reload --host 0.0.0.0 --port 8000
