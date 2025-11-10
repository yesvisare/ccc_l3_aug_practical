# Multi-Agent Orchestration Test Runner
# Runs pytest with proper PYTHONPATH

Write-Host "Running Multi-Agent Orchestration Tests..." -ForegroundColor Green
Write-Host "Setting PYTHONPATH to current directory..." -ForegroundColor Yellow

$env:PYTHONPATH = $PWD

Write-Host "Running pytest..." -ForegroundColor Yellow
python -m pytest tests/ -q

Write-Host "`nTests complete." -ForegroundColor Green
