import os
import time

import optuna
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.logger import TrainingLogger
from src.models import HRM, HREM


class Trainer:
    """
    A flexible trainer class for various models and datasets.
    """

    def __init__(self, model, train_dataset, test_dataset, config):
        self.config = config
        self.training_params = config.get("training", {})

        # Device configuration
        device_str = self.training_params.get("device", "cuda")
        if device_str == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
            device_str = "cpu"
        self.device = torch.device(device_str)
        print(f"Using device: {self.device}")

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model = model.to(self.device)

        # Torch Compile
        if self.training_params.get("use_torch_compile", False):
            try:
                # Check for PyTorch 2.0+
                if hasattr(torch, "compile"):
                    self.model = torch.compile(self.model)
                    print("Model compiled with torch.compile")
                else:
                    print(
                        "Warning: torch.compile not found. "
                        "Requires PyTorch 2.0 or later. "
                        "Continuing without compilation."
                    )
            except Exception as e:
                print(f"Warning: torch.compile failed with error: {e}")
                print("Continuing without compilation.")

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # Performance-tuned DataLoader
        use_pin_memory = self.device.type == "cuda"
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_params.get("batch_size", 32),
            shuffle=True,
            pin_memory=use_pin_memory,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.training_params.get("batch_size", 32),
            pin_memory=use_pin_memory,
        )

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.training_params.get("learning_rate", 0.001)
        )
        self.logger = TrainingLogger(config)

        # --- Log static model info ---
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.log_static_metric("model_num_parameters", num_params)

        # Learning Rate Scheduler
        self.lr_scheduler_params = self.training_params.get("lr_scheduler", {})
        self.lr_scheduler = None
        if self.lr_scheduler_params.get("enabled", False):
            scheduler_type = self.lr_scheduler_params.get("type", "ReduceLROnPlateau")

            if scheduler_type == "ReduceLROnPlateau":
                self.lr_scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode=self.lr_scheduler_params.get("mode", "min"),
                    factor=self.lr_scheduler_params.get("factor", 0.1),
                    patience=self.lr_scheduler_params.get("patience", 10),
                )
            elif scheduler_type == "OneCycleLR":
                self.lr_scheduler = OneCycleLR(
                    self.optimizer,
                    max_lr=self.lr_scheduler_params.get("max_lr", 0.01),
                    steps_per_epoch=len(self.train_loader),
                    epochs=self.training_params.get("epochs", 10),
                )

        # Early Stopping
        self.early_stopping_params = self.training_params.get("early_stopping", {})
        self.early_stopping_patience = self.early_stopping_params.get("patience", 10)
        self.early_stopping_counter = 0
        self.early_stopping_best_loss = float("inf")

        # Gradient Clipping
        self.grad_clip_norm = self.training_params.get("grad_clip_norm", None)

        # Loss function
        if config["dataset"]["name"] == "tiny_shakespeare":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

    def run(self, trial=None):
        epochs = self.training_params.get("epochs", 10)
        epoch_times = []

        for epoch in range(epochs):
            start_time = time.time()
            pbar_desc = f"Epoch {epoch+1}/{epochs}"
            pbar = tqdm(self.train_loader, desc=pbar_desc)

            train_loss = self._train_epoch(pbar)
            test_loss = self._evaluate()

            end_time = time.time()
            epoch_times.append(end_time - start_time)

            self.logger.log(
                {"train_loss": train_loss, "test_loss": test_loss}, step=epoch
            )

            if trial:
                trial.report(test_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if self.lr_scheduler and not isinstance(self.lr_scheduler, OneCycleLR):
                self.lr_scheduler.step(test_loss)

            if test_loss < self.early_stopping_best_loss:
                self.early_stopping_best_loss = test_loss
                if self.training_params.get("checkpointing", False):
                    print(f"New best model saved with loss: {test_loss:.4f}")
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.logger.results_dir, "best_model.pt"),
                    )
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_params.get("enabled", False):
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        # --- Log performance metrics ---
        if epoch_times:
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            self.logger.log_static_metric("avg_epoch_time_s", round(avg_epoch_time, 4))

        self.logger.save_results()
        return self.logger.get_final_metrics()

    def _forward_pass_and_loss(self, batch):
        """
        Performs a forward pass and computes the loss for a given batch.
        Works for both standard and ACT models.
        """
        x, y = batch["inputs"], batch["targets"]
        is_act_model = isinstance(self.model, (HRM, HREM))

        if is_act_model:
            carry = self.model.initial_carry(batch)
            total_loss = 0
            while True:
                if self.model.training:
                    # Detach carry for BPTT.
                    if isinstance(carry, tuple):  # HREM
                        hrm_carry, mem_states = carry
                        if hrm_carry.inner_carry.z_H is not None:
                            hrm_carry.inner_carry.z_H = (
                                hrm_carry.inner_carry.z_H.detach()
                            )
                        if hrm_carry.inner_carry.z_L is not None:
                            hrm_carry.inner_carry.z_L = (
                                hrm_carry.inner_carry.z_L.detach()
                            )
                        carry = (hrm_carry, mem_states)
                    else:  # HRM
                        hrm_carry = carry
                        if hrm_carry.inner_carry.z_H is not None:
                            hrm_carry.inner_carry.z_H = (
                                hrm_carry.inner_carry.z_H.detach()
                            )
                        if hrm_carry.inner_carry.z_L is not None:
                            hrm_carry.inner_carry.z_L = (
                                hrm_carry.inner_carry.z_L.detach()
                            )
                        carry = hrm_carry

                carry, outputs = self.model(carry, batch)
                logits = outputs.get("logits")

                if self.config["dataset"]["name"] == "tiny_shakespeare":
                    task_loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                    )
                else:
                    task_loss = self.criterion(logits, y)

                act_loss = 0
                if "target_q_continue" in outputs:
                    act_loss = torch.nn.BCEWithLogitsLoss()(
                        outputs["q_continue_logits"], outputs["target_q_continue"]
                    )

                total_loss += task_loss + act_loss

                halt_condition = (
                    isinstance(carry, tuple) and carry[0].halted.all()
                ) or (not isinstance(carry, tuple) and carry.halted.all())
                if halt_condition:
                    break
            return total_loss
        else:  # Standard models
            outputs = self.model(batch)
            logits = outputs.get("logits", outputs)
            if self.config["dataset"]["name"] == "tiny_shakespeare":
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                )
            else:
                loss = self.criterion(logits, y)
            return loss

    def _train_epoch(self, pbar):
        self.model.train()
        total_loss = 0

        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            batch = {"inputs": data, "targets": target}

            self.optimizer.zero_grad()

            loss = self._forward_pass_and_loss(batch)

            loss.backward()

            if self.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
            self.optimizer.step()

            if isinstance(self.lr_scheduler, OneCycleLR):
                self.lr_scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        return total_loss / len(self.train_loader)

    def _evaluate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                batch = {"inputs": data, "targets": target}

                loss = self._forward_pass_and_loss(batch)
                total_loss += loss.item()

        return total_loss / len(self.test_loader)
