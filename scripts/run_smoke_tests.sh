#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Smoke Tests ---"

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Project root is one level up from the script directory
PROJECT_ROOT="$SCRIPT_DIR/.."

# Find all smoke test config files and run the experiment for each.
# Use find to be more robust, especially if there are many files.
find "$PROJECT_ROOT/configs" -type f -name '*_smoke.json' | while read config; do
  echo "Running smoke test with config: $config"
  # Run main.py from the project root to ensure correct path resolution
  python "$PROJECT_ROOT/main.py" "$config"
done

echo "--- All smoke tests completed successfully. ---"
