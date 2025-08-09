import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import json
import os
import time
from .datasets import GymnasiumDataset

class Trainer:
    """
    A robust training engine for running experiments.
    Now supports both supervised learning and reinforcement learning.
    """
    def __init__(self, model, train_dataset, test_dataset, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.is_rl_task = isinstance(train_dataset, GymnasiumDataset)

        if not self.is_rl_task:
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

        self.loss_type = self.config['training'].get('loss', 'bce')
        if self.loss_type == 'bce':
            self.loss_fn = nn.BCELoss()
        elif self.loss_type == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.loss_type == 'policy_gradient':
            self.loss_fn = None # Loss is calculated manually
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        self.results = {
            "train_metric": [], # Can be loss or reward
            "eval_metric": 0,
            "best_epoch": 0,
            "training_time": 0
        }
        self.best_metric = -1e9 if self.is_rl_task else -1

        # Create a directory for saving results
        self.exp_dir = os.path.join('results', self.config['experiment_name'])
        os.makedirs(self.exp_dir, exist_ok=True)


    def _train_epoch_sl(self):
        self.model.train()
        total_loss = 0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)

            if isinstance(self.loss_fn, nn.CrossEntropyLoss):
                loss = self.loss_fn(out.view(-1, out.size(-1)), y.view(-1))
            else:
                loss = self.loss_fn(out, y)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _train_epoch_rl(self):
        self.model.train()
        total_rewards = 0
        episodes = self.config['training'].get('episodes_per_epoch', 10)
        gamma = self.config['training'].get('gamma', 0.99)

        for _ in range(episodes):
            state = self.train_dataset.reset()
            done = False
            rewards = []
            log_probs = []

            while not done:
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                action_logits = self.model(state_tensor)
                dist = Categorical(logits=action_logits)
                action = dist.sample()

                log_prob = dist.log_prob(action)
                log_probs.append(log_prob)

                state, reward, done, _ = self.train_dataset.step(action.item())
                rewards.append(reward)

            total_rewards += sum(rewards)

            # Calculate discounted returns
            returns = []
            discounted_reward = 0
            for r in reversed(rewards):
                discounted_reward = r + gamma * discounted_reward
                returns.insert(0, discounted_reward)

            returns = torch.tensor(returns).to(self.device)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Policy gradient loss
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)

            self.optimizer.zero_grad()
            loss = torch.cat(policy_loss).sum()
            loss.backward()
            self.optimizer.step()

        return total_rewards / episodes

    def train_epoch(self):
        if self.is_rl_task:
            return self._train_epoch_rl()
        else:
            return self._train_epoch_sl()

    def _evaluate_sl(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)

                if isinstance(self.loss_fn, nn.CrossEntropyLoss):
                    _, predicted = torch.max(out, 2)
                    correct += (predicted == y).sum().item()
                    total += y.numel()
                else:
                    predicted = (out > 0.5).float()
                    correct += (predicted == y).sum().item()
                    total += y.numel()

        return correct / total

    def _evaluate_rl(self):
        self.model.eval()
        total_rewards = 0
        episodes = self.config['training'].get('eval_episodes', 5)

        for _ in range(episodes):
            state = self.test_dataset.reset()
            done = False
            episode_reward = 0
            while not done:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                    action_logits = self.model(state_tensor)
                    action = torch.argmax(action_logits, dim=-1).item()
                state, reward, done, _ = self.test_dataset.step(action)
                episode_reward += reward
            total_rewards += episode_reward

        return total_rewards / episodes

    def evaluate(self):
        if self.is_rl_task:
            return self._evaluate_rl()
        else:
            return self._evaluate_sl()

    def run(self):
        start_time = time.time()
        print(f"Starting training for experiment: {self.config['experiment_name']}")
        print(f"Using device: {self.device}")

        for epoch in range(self.config['training']['epochs']):
            train_metric = self.train_epoch()
            self.results['train_metric'].append(train_metric)

            eval_metric = self.evaluate()

            metric_name = "Avg Reward" if self.is_rl_task else "Test Acc"
            train_metric_name = "Avg Reward" if self.is_rl_task else "Train Loss"
            print(f"Epoch {epoch+1}/{self.config['training']['epochs']} | "
                  f"{train_metric_name}: {train_metric:.4f} | {metric_name}: {eval_metric:.4f}")

            if eval_metric > self.best_metric:
                self.best_metric = eval_metric
                self.results['best_epoch'] = epoch + 1
                self.results['eval_metric'] = eval_metric
                torch.save(self.model.state_dict(), os.path.join(self.exp_dir, 'best_model.pt'))

        self.results['training_time'] = time.time() - start_time
        metric_name = "Avg Reward" if self.is_rl_task else "Test Accuracy"
        print(f"Training finished. Best {metric_name}: {self.best_metric:.4f}")

        # Save final results and config
        with open(os.path.join(self.exp_dir, 'results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

        return self.results
