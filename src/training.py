import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm

class Trainer:
    """
    A flexible trainer class for various models and datasets.
    """
    def __init__(self, model, train_dataset, test_dataset, config):
        self.config = config
        self.experiment_name = config['experiment_name']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'])

        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['learning_rate'])

        # Loss function depends on the task
        if config['dataset']['name'] == 'tiny_shakespeare':
            # For language modeling, we expect logits and use CrossEntropyLoss
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            # For other tasks, assume reconstruction and use MSE
            self.criterion = torch.nn.MSELoss()

        self.results_dir = f"results/{self.experiment_name}"
        os.makedirs(self.results_dir, exist_ok=True)
        with open(os.path.join(self.results_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def run(self):
        """
        Runs the full training and evaluation loop.
        """
        best_loss = float('inf')
        results = {'train_loss': [], 'test_loss': []}

        for epoch in range(self.config['training']['epochs']):
            # Set a description for the progress bar
            pbar_desc = f"Epoch {epoch+1}/{self.config['training']['epochs']}"

            # Use tqdm for a nice progress bar
            pbar = tqdm(self.train_loader, desc=pbar_desc)

            # Train one epoch
            train_loss = self._train_epoch(pbar)

            # Evaluate on the test set
            test_loss = self._evaluate()

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

            results['train_loss'].append(train_loss)
            results['test_loss'].append(test_loss)

            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(self.model.state_dict(), os.path.join(self.results_dir, 'best_model.pt'))

        # Save final results
        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    def _train_epoch(self, pbar):
        """
        Trains the model for one epoch.
        Handles both standard and ACT-based models.
        """
        self.model.train()
        total_loss = 0

        is_act_model = hasattr(self.model, 'initial_carry')

        # For ACT models, the carry state persists across batches in an epoch.
        carry = None

        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            if is_act_model:
                batch = {'inputs': x, 'targets': y} # Targets might be needed for loss

                # Initialize or detach carry for the new batch
                if carry is None:
                    carry = self.model.initial_carry(batch)
                else:
                    # Detach previous carry state to truncate BPTT
                    hrm_carry, mem_states = carry
                    hrm_carry.inner_carry.z_H = hrm_carry.inner_carry.z_H.detach()
                    hrm_carry.inner_carry.z_L = hrm_carry.inner_carry.z_L.detach()
                    carry = (hrm_carry, mem_states)

                # ACT loop
                total_act_loss = 0
                while True:
                    carry, outputs = self.model(carry, batch)

                    # Calculate task loss
                    if self.config['dataset']['name'] == 'tiny_shakespeare':
                        task_loss = self.criterion(outputs['logits'].view(-1, outputs['logits'].size(-1)), y.view(-1))
                    else:
                        task_loss = self.criterion(outputs['logits'], y)

                    # Calculate ACT loss (if applicable)
                    act_loss = 0
                    if 'target_q_continue' in outputs:
                        q_loss_fn = torch.nn.BCEWithLogitsLoss()
                        act_loss = q_loss_fn(outputs['q_continue_logits'], outputs['target_q_continue'])

                    loss = task_loss + act_loss
                    total_act_loss += loss

                    if carry[0].halted.all():
                        break

                loss = total_act_loss # The final loss is the sum over all ACT steps
            else:
                # Standard model forward pass
                outputs = self.model(x)
                if self.config['dataset']['name'] == 'tiny_shakespeare':
                    loss = self.criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))
                else:
                    loss = self.criterion(outputs, y)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        return total_loss / len(self.train_loader)

    def _evaluate(self):
        """
        Evaluates the model on the test set.
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)

                if self.config['dataset']['name'] == 'tiny_shakespeare':
                    loss = self.criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))
                else:
                    loss = self.criterion(outputs, y)
                total_loss += loss.item()

        return total_loss / len(self.test_loader)
