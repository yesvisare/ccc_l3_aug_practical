# Run FastAPI server for ReAct Pattern Implementation
# Sets PYTHONPATH to current directory and starts uvicorn with reload

$env:PYTHONPATH = $PWD
Write-Host "Starting ReAct Pattern Implementation API..."
Write-Host "PYTHONPATH set to: $env:PYTHONPATH"
Write-Host ""

uvicorn app:app --reload --host 0.0.0.0 --port 8000
