import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the root directory for results and output files
RESULTS_DIR = 'results'
OUTPUT_REPORT_PATH = 'analysis_report.md'
OUTPUT_PLOT_PATH = 'comparison_plot.png'

def load_all_results(results_dir):
    """
    Loads all experiment results from the results directory.
    Returns a list of dictionaries, where each dictionary
    contains the config and results for one experiment.
    """
    all_data = []
    # Project root is the parent directory of the 'scripts' directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_results_dir = os.path.join(project_root, results_dir)

    if not os.path.exists(full_results_dir):
        print(f"Warning: Results directory '{full_results_dir}' not found.")
        return all_data

    for exp_name in sorted(os.listdir(full_results_dir)):
        exp_dir = os.path.join(full_results_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue

        results_path = os.path.join(exp_dir, 'results.json')
        config_path = os.path.join(exp_dir, 'config.json')

        if os.path.exists(results_path) and os.path.exists(config_path):
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                with open(config_path, 'r') as f:
                    config = json.load(f)

                config['experiment_name'] = exp_name
                all_data.append({'config': config, 'results': results})
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON for experiment '{exp_name}'. Skipping.")
        else:
            print(f"Warning: Missing results.json or config.json for '{exp_name}'. Skipping.")

    return all_data

def create_summary_table(all_data):
    """
    Creates a pandas DataFrame summarizing the key results.
    """
    summary_list = []
    for item in all_data:
        config = item['config']
        results = item['results']

        summary = {
            'Experiment': config.get('experiment_name', 'N/A'),
            'Model': config.get('model', {}).get('name', 'N/A'),
            'Dataset': config.get('dataset', {}).get('name', 'N/A'),
            'Test Accuracy': f"{results.get('test_acc', 0):.4f}",
            'Best Epoch': results.get('best_epoch', 'N/A'),
            'Total Epochs': config.get('training', {}).get('epochs', 'N/A'),
            'Training Time (s)': f"{results.get('training_time', 0):.2f}",
        }
        summary_list.append(summary)

    if not summary_list:
        return pd.DataFrame()

    return pd.DataFrame(summary_list)

def create_comparison_plot(all_data, output_path):
    """
    Creates and saves a plot comparing the training loss curves.
    """
    if not all_data:
        print("No data to plot.")
        return

    # Use .get() for safe access to nested keys to prevent KeyErrors
    datasets = sorted(list(set(
        d.get('config', {}).get('dataset', {}).get('name')
        for d in all_data
        if d.get('config', {}).get('dataset', {}).get('name') is not None
    )))

    num_datasets = len(datasets)
    if num_datasets == 0:
        print("No valid datasets found to plot.")
        return

    fig, axes = plt.subplots(num_datasets, 1, figsize=(12, 7 * num_datasets), squeeze=False)
    fig.suptitle('Training Loss Comparison', fontsize=16, y=0.98)

    for i, dataset_name in enumerate(datasets):
        ax = axes[i, 0]
        sns.set_style("whitegrid")

        dataset_data = [d for d in all_data if d.get('config', {}).get('dataset', {}).get('name') == dataset_name]

        for item in dataset_data:
            exp_name = item['config']['experiment_name']
            loss_curve = item['results'].get('train_loss', [])
            if loss_curve:
                ax.plot(range(1, len(loss_curve) + 1), loss_curve, label=exp_name)

        ax.set_title(f'Task: {dataset_name}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss (Log Scale)')
        ax.legend(loc='best')
        ax.set_yscale('log')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Comparison plot saved to {output_path}")

def generate_conclusion(summary_df):
    """
    Generates a textual conclusion based on the summary DataFrame.
    """
    if summary_df.empty:
        return "## Analysis Conclusion\n\nNo results were available to analyze."

    conclusion = "## Analysis Conclusion\n\n"
    conclusion += "This report summarizes the performance of various models across different tasks. "

    df_copy = summary_df.copy()
    df_copy['Test Accuracy'] = pd.to_numeric(df_copy['Test Accuracy'], errors='coerce')

    if not df_copy['Test Accuracy'].isnull().all():
        best_overall = df_copy.loc[df_copy['Test Accuracy'].idxmax()]
        conclusion += (f"The top-performing model overall was **{best_overall['Experiment']}** "
                       f"on the **{best_overall['Dataset']}** task, achieving a test accuracy of "
                       f"**{best_overall['Test Accuracy']:.4f}**.\n\n")
    else:
        conclusion += "Accuracy metrics were not available for a conclusive performance ranking.\n\n"

    for dataset, group in df_copy.groupby('Dataset'):
        if group.shape[0] > 1 and not group['Test Accuracy'].isnull().all():
            best_in_group = group.loc[group['Test Accuracy'].idxmax()]
            conclusion += (f"For the **{dataset}** task, **{best_in_group['Experiment']}** was the most effective model. "
                           f"This suggests its architecture may be particularly well-suited for this problem.\n")

    conclusion += "\nThe accompanying plot of training curves provides further insight into the learning dynamics. "
    conclusion += "Models that achieve a lower final loss more quickly are generally preferable. "
    conclusion += "These initial findings can guide further research, such as more extensive hyperparameter tuning on the most promising models."

    return conclusion

def main():
    """
    Main function to run the analysis.
    """
    print("--- Starting Analysis ---")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    all_data = load_all_results(RESULTS_DIR)

    report_path = os.path.join(project_root, OUTPUT_REPORT_PATH)
    plot_path = os.path.join(project_root, OUTPUT_PLOT_PATH)

    if not all_data:
        print("No valid experiment results found. Exiting analysis.")
        with open(report_path, 'w') as f:
            f.write("# Analysis Report\n\n")
            f.write("No valid experiment results found in the 'results' directory.\n")
        return

    summary_df = create_summary_table(all_data)
    create_comparison_plot(all_data, plot_path)
    conclusion_text = generate_conclusion(summary_df)

    with open(report_path, 'w') as f:
        f.write("# Experiment Analysis Report\n\n")
        f.write("This report provides a comparative analysis of the conducted experiments.\n\n")
        f.write("## Summary of Results\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        f.write(f"![Training Loss Comparison]({os.path.basename(plot_path)})\n\n")
        f.write(conclusion_text)

    print(f"--- Analysis complete. Report saved to {report_path} ---")

if __name__ == '__main__':
    main()
