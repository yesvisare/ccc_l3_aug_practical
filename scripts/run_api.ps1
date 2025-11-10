# PowerShell script to run the FastAPI server
# Windows-first approach for L3 baseline

$env:PYTHONPATH = $PWD
Write-Host "Starting Conversational RAG API server..."
Write-Host "PYTHONPATH set to: $env:PYTHONPATH"
Write-Host ""

uvicorn app:app --reload --host 0.0.0.0 --port 8000
