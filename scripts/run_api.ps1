# PowerShell script to run the API server
# Sets PYTHONPATH and starts uvicorn with reload

$env:PYTHONPATH = $PWD
Write-Host "Starting API server..."
Write-Host "PYTHONPATH: $env:PYTHONPATH"
uvicorn app:app --reload --host 0.0.0.0 --port 8000
