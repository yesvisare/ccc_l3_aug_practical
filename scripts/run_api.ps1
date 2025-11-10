# Multi-Agent Orchestration API Server
# Runs FastAPI server with auto-reload

Write-Host "Starting Multi-Agent Orchestration API..." -ForegroundColor Green
Write-Host "Setting PYTHONPATH to current directory..." -ForegroundColor Yellow

$env:PYTHONPATH = $PWD

Write-Host "Launching uvicorn server..." -ForegroundColor Yellow
uvicorn app:app --reload --host 0.0.0.0 --port 8000

Write-Host "`nAPI Server stopped." -ForegroundColor Red
