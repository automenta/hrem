import os
import sys
import argparse

# Add the project root to the Python path to allow importing from 'src' and 'main'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from main import get_model, get_dataset, load_config  # noqa: E402


def validate_all_configs(config_dir):
    """
    Validates all .json configuration files in a given directory.
    """
    print(f"--- Starting Configuration Validation in {config_dir} ---")
    config_files = sorted([f for f in os.listdir(config_dir) if f.endswith(".json")])
    invalid_configs = []

    for config_file in config_files:
        config_path = os.path.join(config_dir, config_file)
        try:
            print(f"Validating {config_file}...")
            config = load_config(config_path)

            # get_dataset can modify the config in-place, which is necessary
            # for some models (e.g., setting input_size).
            get_dataset(config)

            # get_model instantiates the model, which triggers Pydantic validation.
            get_model(config)

            print(f"  \u2713 {config_file} is valid.")

        except Exception as e:
            print(
                f"  \u2717 Error validating {config_file}: {e.__class__.__name__}: {e}"
            )
            invalid_configs.append(config_file)

    return invalid_configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate model configuration files.")
    parser.add_argument(
        "--config_dir",
        type=str,
        default=os.path.join(project_root, "configs"),
        help="Directory containing the configuration files to validate.",
    )
    args = parser.parse_args()

    invalid_files = validate_all_configs(args.config_dir)

    if invalid_files:
        print(f"\nValidation FAILED for {len(invalid_files)} file(s):")
        for f in invalid_files:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\n--- All configuration files are valid. ---")
        sys.exit(0)
