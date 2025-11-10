# Run FastAPI server with auto-reload
# Usage: .\scripts\run_api.ps1

$env:PYTHONPATH = $PWD
uvicorn app:app --reload --host 127.0.0.1 --port 8000
