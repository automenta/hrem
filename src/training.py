import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
from tqdm import tqdm
import numpy as np
from src.models import HRM, HREM

class Trainer:
    """
    A flexible trainer class for various models and datasets, now with advanced features.
    """
    def __init__(self, model, train_dataset, test_dataset, config):
        self.config = config
        self.experiment_name = config['experiment_name']
        self.training_params = config.get('training', {})

        # Device configuration
        device_str = self.training_params.get('device', 'cuda')
        if device_str == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
            device_str = 'cpu'
        self.device = torch.device(device_str)
        print(f"Using device: {self.device}")

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # Performance-tuned DataLoader
        use_pin_memory = self.device.type == 'cuda'
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_params.get('batch_size', 32),
            shuffle=True,
            pin_memory=use_pin_memory
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.training_params.get('batch_size', 32),
            pin_memory=use_pin_memory
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.training_params.get('learning_rate', 0.001))

        # Learning Rate Scheduler
        self.lr_scheduler_params = self.training_params.get('lr_scheduler', {})
        if self.lr_scheduler_params.get('enabled', False):
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=self.lr_scheduler_params.get('mode', 'min'),
                factor=self.lr_scheduler_params.get('factor', 0.1),
                patience=self.lr_scheduler_params.get('patience', 10)
            )
        else:
            self.lr_scheduler = None

        # Early Stopping
        self.early_stopping_params = self.training_params.get('early_stopping', {})
        self.early_stopping_patience = self.early_stopping_params.get('patience', 10)
        self.early_stopping_counter = 0
        self.early_stopping_best_loss = float('inf')

        # Gradient Clipping
        self.grad_clip_norm = self.training_params.get('grad_clip_norm', None)

        # Loss function
        if config['dataset']['name'] == 'tiny_shakespeare':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

        self.results_dir = f"results/{self.experiment_name}"
        os.makedirs(self.results_dir, exist_ok=True)
        with open(os.path.join(self.results_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def run(self):
        results = {'train_loss': [], 'test_loss': []}
        epochs = self.training_params.get('epochs', 10)

        for epoch in range(epochs):
            pbar_desc = f"Epoch {epoch+1}/{epochs}"
            pbar = tqdm(self.train_loader, desc=pbar_desc)

            train_loss = self._train_epoch(pbar)
            test_loss = self._evaluate()

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

            results['train_loss'].append(train_loss)
            results['test_loss'].append(test_loss)

            if self.lr_scheduler:
                self.lr_scheduler.step(test_loss)

            if test_loss < self.early_stopping_best_loss:
                self.early_stopping_best_loss = test_loss
                print(f"New best model saved with loss: {test_loss:.4f}")
                torch.save(self.model.state_dict(), os.path.join(self.results_dir, 'best_model.pt'))
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_params.get('enabled', False):
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    def _train_epoch(self, pbar):
        self.model.train()
        total_loss = 0
        is_act_model = isinstance(self.model, (HRM, HREM))

        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            batch = {'inputs': x, 'targets': y}
            self.optimizer.zero_grad()

            if is_act_model:
                carry = self.model.initial_carry(batch)
                batch_total_loss = 0

                while True:
                    # Detach carry for BPTT. Handle both HRM and HREM carry structures.
                    if isinstance(carry, tuple): # HREM
                        hrm_carry, mem_states = carry
                        if hrm_carry.inner_carry.z_H is not None: hrm_carry.inner_carry.z_H = hrm_carry.inner_carry.z_H.detach()
                        if hrm_carry.inner_carry.z_L is not None: hrm_carry.inner_carry.z_L = hrm_carry.inner_carry.z_L.detach()
                        carry = (hrm_carry, mem_states)
                    else: # HRM
                        hrm_carry = carry
                        if hrm_carry.inner_carry.z_H is not None: hrm_carry.inner_carry.z_H = hrm_carry.inner_carry.z_H.detach()
                        if hrm_carry.inner_carry.z_L is not None: hrm_carry.inner_carry.z_L = hrm_carry.inner_carry.z_L.detach()
                        carry = hrm_carry

                    carry, outputs = self.model(carry, batch)

                    logits = outputs.get('logits')
                    if self.config['dataset']['name'] == 'tiny_shakespeare':
                        task_loss = self.criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    else:
                        task_loss = self.criterion(logits, y)

                    act_loss = 0
                    if 'target_q_continue' in outputs:
                        act_loss = torch.nn.BCEWithLogitsLoss()(outputs['q_continue_logits'], outputs['target_q_continue'])

                    batch_total_loss += task_loss + act_loss

                    if (isinstance(carry, tuple) and carry[0].halted.all()) or \
                       (not isinstance(carry, tuple) and carry.halted.all()):
                        break

                loss = batch_total_loss
                loss.backward()
                final_loss_val = loss.item()

            else: # Standard models
                outputs = self.model(batch)
                logits = outputs.get('logits', outputs)
                if self.config['dataset']['name'] == 'tiny_shakespeare':
                    loss = self.criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                else:
                    loss = self.criterion(logits, y)
                loss.backward()
                final_loss_val = loss.item()

            if self.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            total_loss += final_loss_val
            pbar.set_postfix({'loss': final_loss_val})

        return total_loss / len(self.train_loader)

    def _evaluate(self):
        self.model.eval()
        total_loss = 0
        is_act_model = isinstance(self.model, (HRM, HREM))

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                batch = {'inputs': x, 'targets': y}

                if is_act_model:
                    carry = self.model.initial_carry(batch)
                    batch_total_loss = 0
                    while True:
                        carry, outputs = self.model(carry, batch)
                        logits = outputs.get('logits')
                        if self.config['dataset']['name'] == 'tiny_shakespeare':
                            task_loss = self.criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                        else:
                            task_loss = self.criterion(logits, y)

                        act_loss = 0
                        if 'target_q_continue' in outputs:
                            act_loss = torch.nn.BCEWithLogitsLoss()(outputs['q_continue_logits'], outputs['target_q_continue'])

                        batch_total_loss += (task_loss + act_loss).item()

                        if (isinstance(carry, tuple) and carry[0].halted.all()) or \
                           (not isinstance(carry, tuple) and carry.halted.all()):
                            break
                    final_loss_val = batch_total_loss
                else:
                    outputs = self.model(batch)
                    logits = outputs.get('logits', outputs)
                    if self.config['dataset']['name'] == 'tiny_shakespeare':
                        loss = self.criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    else:
                        loss = self.criterion(logits, y)
                    final_loss_val = loss.item()

                total_loss += final_loss_val
        return total_loss / len(self.test_loader)
