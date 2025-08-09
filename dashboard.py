import streamlit as st
import os
import pandas as pd
import json
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="HREM Evaluation Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- Helper Functions ---
@st.cache_data
def load_all_results_data(results_dir="results"):
    """
    Loads all experiment results from the results directory.
    Returns a list of dictionaries, where each dictionary
    contains the config and results for one experiment.
    """
    all_data = []
    if not os.path.exists(results_dir):
        st.warning(f"Results directory '{results_dir}' not found.")
        return all_data, pd.DataFrame()

    exp_names = sorted([d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))])

    summary_list = []
    for exp_name in exp_names:
        exp_dir = os.path.join(results_dir, exp_name)
        results_path = os.path.join(exp_dir, 'results.json')
        config_path = os.path.join(exp_dir, 'config.json')

        if os.path.exists(results_path) and os.path.exists(config_path):
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                with open(config_path, 'r') as f:
                    config = json.load(f)

                all_data.append({'name': exp_name, 'config': config, 'results': results})

                summary = {
                    'Experiment': exp_name,
                    'Model': config.get('model', {}).get('name', 'N/A'),
                    'Dataset': config.get('dataset', {}).get('name', 'N/A'),
                    'Test Loss': f"{min(results.get('test_loss', [0])):.4f}",
                    'Best Epoch': results.get('best_epoch', 'N/A'),
                    'Total Epochs': config.get('training', {}).get('epochs', 'N/A'),
                }
                summary_list.append(summary)
            except (json.JSONDecodeError, KeyError) as e:
                st.warning(f"Could not load results for '{exp_name}': {e}")

    summary_df = pd.DataFrame(summary_list)
    return all_data, summary_df

def create_interactive_plot(selected_data):
    """Creates an interactive Plotly chart for comparing experiments."""
    plot_df_list = []
    for item in selected_data:
        loss_data = {
            'train': item['results'].get('train_loss', []),
            'test': item['results'].get('test_loss', [])
        }
        for loss_type, loss_values in loss_data.items():
            for epoch, loss in enumerate(loss_values):
                plot_df_list.append({
                    'Experiment': item['name'],
                    'Epoch': epoch + 1,
                    'Loss': loss,
                    'Loss Type': loss_type
                })

    if not plot_df_list:
        return None

    plot_df = pd.DataFrame(plot_df_list)

    fig = px.line(
        plot_df,
        x='Epoch',
        y='Loss',
        color='Experiment',
        line_dash='Loss Type',
        title='Training and Test Loss Comparison',
        labels={'Loss': 'Loss (Log Scale)', 'Epoch': 'Epoch'},
        log_y=True
    )
    fig.update_layout(legend_title_text='Legend')
    return fig

# --- Main Dashboard Page ---
def main_dashboard():
    st.title("📊 HREM Evaluation Dashboard")
    st.write("Browse and compare the results of all completed experiments.")

    all_data, summary_df = load_all_results_data()

    if not all_data:
        st.warning("No experiments found. Run some tests or training sessions first!")
        return

    # --- Display Summary Table ---
    st.header("📋 Experiment Summary")
    st.dataframe(summary_df, use_container_width=True)

    # --- Experiment Comparison ---
    st.header("🔬 Compare Experiments")

    exp_names = [d['name'] for d in all_data]
    default_selection = [exp_names[0]] if exp_names else []

    selected_exps = st.multiselect(
        "Select experiments to compare:",
        options=exp_names,
        default=default_selection
    )

    if not selected_exps:
        st.info("Select one or more experiments to visualize their loss curves.")
        return

    selected_data = [d for d in all_data if d['name'] in selected_exps]

    # --- Display Plots and Configs ---
    st.subheader("📈 Loss Curves")
    fig = create_interactive_plot(selected_data)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No loss data available for the selected experiments.")

    st.subheader("⚙️ Configurations")
    for item in selected_data:
        with st.expander(f"View config for `{item['name']}`"):
            st.json(item['config'])

# --- App ---
if __name__ == "__main__":
    main_dashboard()
