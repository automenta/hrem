import streamlit as st
import os
import pandas as pd
from src.analysis import load_results, plot_training_curve

# --- Page Configuration ---
st.set_page_config(
    page_title="HREM Evaluation Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- Helper Functions ---
@st.cache_data
def find_experiments(results_dir="results"):
    """Finds all experiment subdirectories in the results folder."""
    if not os.path.exists(results_dir):
        return []
    return [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

# --- Main Dashboard Page ---
def main_dashboard():
    st.title("📊 Experiment Dashboard")
    st.write("Browse and visualize the results of completed experiments.")

    exp_dirs = find_experiments()
    if not exp_dirs:
        st.warning("No experiments found. Run some tests or training sessions first!")
        return

    # --- Experiment Selection ---
    selected_exp = st.selectbox("Select an experiment to view:", exp_dirs)

    if selected_exp:
        exp_path = os.path.join("results", selected_exp)

        try:
            results, config = load_results(exp_path)
        except FileNotFoundError:
            st.error(f"Could not load results for '{selected_exp}'. The results/config file may be missing.")
            return

        st.header(f"Results for: `{selected_exp}`")

        # --- Display Metrics ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Test Accuracy", f"{results.get('test_acc', 0):.4f}")
        col2.metric("Best Epoch", f"{results.get('best_epoch', 0)}")
        col3.metric("Training Time", f"{results.get('training_time', 0):.2f}s")

        # --- Display Plots and Config ---
        tab1, tab2 = st.tabs(["📈 Training Curve", "⚙️ Configuration"])

        with tab1:
            plot_path = os.path.join(exp_path, 'training_curve.png')
            if not os.path.exists(plot_path):
                # Generate it if it doesn't exist
                plot_training_curve(results, save_path=plot_path)

            if os.path.exists(plot_path):
                st.image(plot_path, caption="Training Loss over Epochs")
            else:
                st.warning("Training curve plot not found.")

        with tab2:
            st.json(config)


# --- App ---
if __name__ == "__main__":
    main_dashboard()
