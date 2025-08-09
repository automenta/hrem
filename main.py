import argparse
import json
import torch.nn as nn
from src.models import MLP, HRM, HREM
from src.datasets import ReverseDataset, TinyShakespeareDataset, CopyTaskDataset, AssociativeRecallDataset, GymnasiumDataset
from src.training import Trainer

def main(config_path):
    """
    Main function to run an experiment.
    """
    # 1. Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 2. Select dataset
    dataset_name = config['dataset']['name']
    dataset_params = config['dataset']['params']
    model_params = config['model']['params']

    if dataset_name == 'reverse':
        train_ds = ReverseDataset(dataset_params['train_size'], dataset_params['seq_len'])
        test_ds = ReverseDataset(dataset_params['test_size'], dataset_params['seq_len'])
    elif dataset_name == 'tiny_shakespeare':
        train_ds = TinyShakespeareDataset(dataset_params['seq_length'], split='train')
        test_ds = TinyShakespeareDataset(dataset_params['seq_length'], split='test')
    elif dataset_name == 'copy':
        train_ds = CopyTaskDataset(size=dataset_params['train_size'], seq_len=dataset_params['seq_len'], vec_len=dataset_params['vec_len'])
        test_ds = CopyTaskDataset(size=dataset_params['test_size'], seq_len=dataset_params['seq_len'], vec_len=dataset_params['vec_len'])
    elif dataset_name == 'associative_recall':
        train_ds = AssociativeRecallDataset(size=dataset_params['train_size'], item_range=tuple(dataset_params['item_range']), vec_len=dataset_params['vec_len'])
        test_ds = AssociativeRecallDataset(size=dataset_params['test_size'], item_range=tuple(dataset_params['item_range']), vec_len=dataset_params['vec_len'])
    elif dataset_name == 'gymnasium':
        # For RL, train and test datasets are environment wrappers
        train_ds = GymnasiumDataset(env_name=dataset_params['env_name'])
        test_ds = GymnasiumDataset(env_name=dataset_params['env_name'])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 3. Select model
    model_name = config['model']['name']
    if model_name == 'mlp':
        model = MLP(**model_params)
    elif model_name == 'hrm':
        model = HRM(**model_params)
    elif model_name == 'hrem':
        model = HREM(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 4. Adapt model for RL tasks if necessary
    if isinstance(train_ds, GymnasiumDataset):
        # The output layer must match the number of actions
        if hasattr(model, 'f_O'):
            model.f_O = nn.Linear(model.d_model, train_ds.action_space.n)
        elif hasattr(model, 'fc3'): # For MLP
            model.fc3 = nn.Linear(model.fc3.in_features, train_ds.action_space.n)

    # 5. Instantiate and run the trainer
    trainer = Trainer(model, train_ds, test_ds, config)
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a training experiment.")
    parser.add_argument('config', type=str, help='Path to the JSON configuration file.')
    args = parser.parse_args()

    main(args.config)
