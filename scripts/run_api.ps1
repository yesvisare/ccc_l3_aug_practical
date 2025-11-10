# Run FastAPI application
# Sets PYTHONPATH to current directory and starts uvicorn with reload

$env:PYTHONPATH = $PWD
uvicorn app:app --reload --host 0.0.0.0 --port 8000
