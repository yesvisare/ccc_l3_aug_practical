# Run API Server
# Usage: .\scripts\run_api.ps1

$env:PYTHONPATH = $PWD
uvicorn app:app --reload --host 0.0.0.0 --port 8000
