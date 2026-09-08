import os
import torch
import torch.nn as nn
from src.training.config import TrainingConfig

class Trainer:
    # master training loop with early stopping and checkpointing
    def __init__(
            self,
            model: nn.Module,
            loss_fn: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
            config: TrainingConfig,
            device: torch.device,
            save_dir: str = "checkpoints"
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.save_dir = save_dir

        self.scalar = torch.amp.GradScaler(device.type, enabled=self.config.use_amp)

        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def _train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for batch_idx, (x, t_true, s_true) in enumerate(dataloader):
            x = x.to(self.device)
            t_true = t_true.to(self.device)
            s_true = s_true.to(self.device)

            # fwd + amp
            with torch.amp.autocast(self.device.type, enabled = self.config.use_amp):
                t_pred, s_pred = self.model(x)
                loss, _ = self.loss_fn(t_pred, s_pred, t_true, s_true)
                loss = loss / self.config.accumulation_steps 

            # bwd
            self.scalar.scale(loss).backward()

            # accumulation steps
            if (batch_idx+1) % self.config.accumulation_steps == 0 or (batch_idx+1) == len(dataloader):
                self.scalar.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norms)

                self.scalar.step(self.optimizer)
                self.scalar.update()
                self.optimizer.zero_grad()

            total_loss += (loss.item() * self.config.accumulation_steps)

        return total_loss / len(dataloader)

    @torch.no_grad()
    def _val_epoch(self, dataloader) -> float:
        self.model.eval()
        total_loss = 0.0

        for x, t_true, s_true in dataloader:
            x, t_true, s_true = x.to(self.device), t_true.to(self.device), s_true.to(self.device)

            with torch.amp.autocast(self.device.type, enabled=self.config.use_amp):
                t_pred, s_pred = self.model(x)
                loss, _ = self.loss_fn(t_pred, s_pred, t_true, s_true)

            total_loss += loss.item()
        return total_loss / len(dataloader)

    def fit(self, train_loader, val_loader):
        print(f"Starting Training for up to {self.config.max_epochs} epochs.")
        checkpoint_path = os.path.join(self.save_dir, "best_model.pt")

        for epoch in range(self.config.max_epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._val_epoch(val_loader)

            # step scheduler based on validation loss
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # early stopping and checkpointing logic
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss
                }, checkpoint_path)
                print(f"[+] New best model saved to {checkpoint_path}")
            else:
                self.early_stop_counter += 1
                print(f"[-] No improvement. Early stop counter: {self.early_stop_counter}/{self.config.early_stopping_patience}")

            if self.early_stop_counter >= self.config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        print("Restoring best model weights")
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])