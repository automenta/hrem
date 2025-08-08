import argparse
import json
from src.models import MLP, HRM, HREM
from src.datasets import ReverseDataset, TinyShakespeareDataset
from src.training import Trainer

def main(config_path):
    """
    Main function to run an experiment.
    """
    # 1. Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 2. Select model
    model_name = config['model']['name']
    model_params = config['model']['params']
    if model_name == 'mlp':
        model = MLP(**model_params)
    elif model_name == 'hrm':
        model = HRM(**model_params)
    elif model_name == 'hrem':
        model = HREM(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 3. Select dataset
    dataset_name = config['dataset']['name']
    dataset_params = config['dataset']['params']
    if dataset_name == 'reverse':
        train_ds = ReverseDataset(dataset_params['train_size'], dataset_params['seq_len'])
        test_ds = ReverseDataset(dataset_params['test_size'], dataset_params['seq_len'])
    elif dataset_name == 'tiny_shakespeare':
        train_ds = TinyShakespeareDataset(dataset_params['seq_length'], split='train')
        test_ds = TinyShakespeareDataset(dataset_params['seq_length'], split='test')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 4. Instantiate and run the trainer
    trainer = Trainer(model, train_ds, test_ds, config)
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a training experiment.")
    parser.add_argument('config', type=str, help='Path to the JSON configuration file.')
    args = parser.parse_args()

    main(args.config)
