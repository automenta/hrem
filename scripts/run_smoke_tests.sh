#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Smoke Tests ---"

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Project root is one level up from the script directory
export PROJECT_ROOT="$SCRIPT_DIR/.."

# Get the number of available processors
if [[ "$(uname)" == "Darwin" ]]; then
    NPROC=$(sysctl -n hw.ncpu)
else
    NPROC=$(nproc)
fi

echo "Running smoke tests in parallel with $NPROC processes..."

# Find all smoke test config files and run the experiment for each in parallel.
# The -I flag implies one item per command, so -n 1 is not needed.
# We must export PROJECT_ROOT so it's available in the subshell created by bash -c.
find "$PROJECT_ROOT/configs" -type f -name '*_smoke.json' -print0 | xargs -0 -P "$NPROC" -I {} bash -c '
    echo "Running smoke test with config: {}"
    python "$PROJECT_ROOT/main.py" "{}"
'

echo "--- All smoke tests completed successfully. ---"
