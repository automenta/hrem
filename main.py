import argparse
import json
import torch.nn as nn
from src.models import MLP, HREM
from src.datasets import ReverseDataset, TinyShakespeareDataset, CopyTaskDataset, AssociativeRecallDataset, GymnasiumDataset
from src.training import Trainer

def main(config_path):
    """
    Main function to run an experiment.
    """
    # 1. Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 2. Select and instantiate dataset
    dataset_name = config['dataset']['name']
    dataset_params = config['dataset']['params']

    if dataset_name == 'reverse':
        train_ds = ReverseDataset(dataset_params['train_size'], dataset_params['seq_len'])
        test_ds = ReverseDataset(dataset_params['test_size'], dataset_params['seq_len'])
        # For sequence models, input_size is the feature dimension at each step
        config['model']['params']['input_size'] = 1 # Binary classification at each step
    elif dataset_name == 'tiny_shakespeare':
        train_ds = TinyShakespeareDataset(seq_length=dataset_params['seq_length'], split='train')
        test_ds = TinyShakespeareDataset(seq_length=dataset_params['seq_length'], split='test')
        config['model']['params']['vocab_size'] = train_ds.vocab_size
        config['model']['params']['input_size'] = train_ds.vocab_size # Not used by HREM if vocab_size is set
    elif dataset_name == 'copy':
        train_ds = CopyTaskDataset(size=dataset_params['train_size'], seq_len=dataset_params['seq_len'], vec_len=dataset_params['vec_len'])
        test_ds = CopyTaskDataset(size=dataset_params['test_size'], seq_len=dataset_params['seq_len'], vec_len=dataset_params['vec_len'])
        config['model']['params']['input_size'] = dataset_params['vec_len']
    elif dataset_name == 'associative_recall':
        train_ds = AssociativeRecallDataset(size=dataset_params['train_size'], item_range=tuple(dataset_params['item_range']), vec_len=dataset_params['vec_len'])
        test_ds = AssociativeRecallDataset(size=dataset_params['test_size'], item_range=tuple(dataset_params['item_range']), vec_len=dataset_params['vec_len'])
        config['model']['params']['input_size'] = dataset_params['vec_len']
    elif dataset_name == 'gymnasium':
        train_ds = GymnasiumDataset(env_name=dataset_params['env_name'])
        test_ds = GymnasiumDataset(env_name=dataset_params['env_name'])
        # This part will need to be revisited if RL is brought back
        config['model']['params']['input_size'] = train_ds.obs_space.shape[0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 3. Select and instantiate model
    model_name = config['model']['name']
    model_params = config['model']['params']

    if model_name == 'mlp':
        # MLP needs a flattened input size
        if dataset_name == 'copy':
             model_params['input_size'] = (dataset_params['seq_len'] * 2 + 1) * dataset_params['vec_len']
             model_params['output_size'] = model_params['input_size']
        elif dataset_name == 'associative_recall':
            model_params['input_size'] = (dataset_params['item_range'][1] + 2) * dataset_params['vec_len']
            model_params['output_size'] = model_params['input_size']
        # Add other MLP cases if needed
        model = MLP(**model_params)
    elif model_name == 'hrem':
        model = HREM(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 4. Instantiate and run the trainer
    trainer = Trainer(model, train_ds, test_ds, config)
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a training experiment.")
    parser.add_argument('config', type=str, help='Path to the JSON configuration file.')
    args = parser.parse_args()

    main(args.config)
