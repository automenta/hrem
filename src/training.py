import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import time

class Trainer:
    """
    A robust training engine for running experiments.
    """
    def __init__(self, model, train_dataset, test_dataset, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['training']['batch_size']
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )

        loss_type = self.config['training'].get('loss', 'bce')
        if loss_type == 'bce':
            self.loss_fn = nn.BCELoss()
        elif loss_type == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        self.results = {
            "train_loss": [],
            "test_acc": 0,
            "best_epoch": 0,
            "training_time": 0
        }
        self.best_metric = -1

        # Create a directory for saving results
        self.exp_dir = os.path.join('results', self.config['experiment_name'])
        os.makedirs(self.exp_dir, exist_ok=True)


    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)

            # Adjust for loss function requirements
            if isinstance(self.loss_fn, nn.CrossEntropyLoss):
                # CrossEntropyLoss expects logits, and target shape (N) or (N, d1, d2, ...)
                # Output shape: (batch_size, seq_len, vocab_size), Target shape: (batch_size, seq_len)
                loss = self.loss_fn(out.view(-1, out.size(-1)), y.view(-1))
            else:
                loss = self.loss_fn(out, y)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)

                if isinstance(self.loss_fn, nn.CrossEntropyLoss):
                    # For language model, accuracy is predicting the next character
                    _, predicted = torch.max(out, 2)
                    correct += (predicted == y).sum().item()
                    total += y.numel()
                else:
                    # For reversal task, accuracy is based on 0.5 threshold
                    predicted = (out > 0.5).float()
                    correct += (predicted == y).sum().item()
                    total += y.numel()

        return correct / total

    def run(self):
        start_time = time.time()
        print(f"Starting training for experiment: {self.config['experiment_name']}")
        print(f"Using device: {self.device}")

        for epoch in range(self.config['training']['epochs']):
            train_loss = self.train_epoch()
            self.results['train_loss'].append(train_loss)

            test_acc = self.evaluate()

            print(f"Epoch {epoch+1}/{self.config['training']['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}")

            if test_acc > self.best_metric:
                self.best_metric = test_acc
                self.results['best_epoch'] = epoch + 1
                self.results['test_acc'] = test_acc
                # Save the best model
                torch.save(self.model.state_dict(), os.path.join(self.exp_dir, 'best_model.pt'))

        self.results['training_time'] = time.time() - start_time
        print(f"Training finished. Best test accuracy: {self.best_metric:.4f}")

        # Save final results and config
        with open(os.path.join(self.exp_dir, 'results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

        return self.results
