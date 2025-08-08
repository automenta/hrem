import json
import os
import matplotlib.pyplot as plt

def load_results(exp_dir):
    """Loads results and config from a given experiment directory."""
    results_path = os.path.join(exp_dir, 'results.json')
    config_path = os.path.join(exp_dir, 'config.json')

    if not os.path.exists(results_path) or not os.path.exists(config_path):
        raise FileNotFoundError(f"Results or config file not found in {exp_dir}")

    with open(results_path, 'r') as f:
        results = json.load(f)
    with open(config_path, 'r') as f:
        config = json.load(f)

    return results, config

def plot_training_curve(results, save_path=None):
    """Plots and saves the training loss curve."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots()
    ax.plot(range(1, len(results['train_loss']) + 1), results['train_loss'])
    ax.set_title('Training Loss Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)

def generate_report(exp_dir):
    """
    Loads data from an experiment and generates a simple text and plot report.
    """
    print(f"--- Generating Report for {os.path.basename(exp_dir)} ---")

    try:
        results, config = load_results(exp_dir)
    except FileNotFoundError as e:
        print(e)
        return

    print("\n[Configuration]")
    print(json.dumps(config, indent=2))

    print("\n[Results]")
    print(f"  - Best Test Accuracy: {results.get('test_acc', 'N/A'):.4f}")
    print(f"  - Achieved at Epoch: {results.get('best_epoch', 'N/A')}")
    print(f"  - Total Training Time: {results.get('training_time', 'N/A'):.2f}s")

    # Generate and save the training curve plot
    plot_path = os.path.join(exp_dir, 'training_curve.png')
    plot_training_curve(results, save_path=plot_path)

    print("\nReport generation complete.")
