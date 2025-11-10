# Run tests for Module 12.4
# Tenant Lifecycle Management Tests

Write-Host "Running Module 12.4 tests..." -ForegroundColor Green

# Set PYTHONPATH to include project root
$env:PYTHONPATH = $PWD

# Run pytest with quiet mode
pytest -q tests/

# For verbose output, use:
# pytest -v tests/

# For specific test file:
# pytest -v tests/test_m12_tenant_lifecycle_management.py
