#!/bin/bash
# Run tests for Module 12.4
# Tenant Lifecycle Management Tests

echo "Running Module 12.4 tests..."

# Set PYTHONPATH to include project root
export PYTHONPATH=$(pwd)

# Run pytest with quiet mode
pytest -q tests/

# For verbose output, use:
# pytest -v tests/

# For specific test file:
# pytest -v tests/test_m12_tenant_lifecycle_management.py
