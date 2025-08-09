#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Full Experiments ---"

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Project root is one level up from the script directory
PROJECT_ROOT="$SCRIPT_DIR/.."

# Find all non-smoke test config files and run the experiment for each.
find "$PROJECT_ROOT/configs" -type f -name '*.json' ! -name '*_smoke.json' ! -name 'default.json' | while read config; do
  echo "Running full experiment with config: $config"
  # Run main.py from the project root
  python "$PROJECT_ROOT/main.py" "$config"
done

echo "--- All full experiments completed successfully. ---"
