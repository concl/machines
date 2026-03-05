from template import *
from dataclasses import dataclass


class Trainer:
    """
    A class for training a PyTorch model
    """

    def __init__(self, model, args=None):

        self.model = model
        self.args = args or TrainingArguments()

    def train(self, train_loader, valid_loader=None):

        device = self.args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        optimizer = self.args.optimizer_class(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            **(self.args.optimizer_kwargs or {})
        )
        criterion = self.args.criterion
        scheduler = self.args.scheduler(optimizer, patience=3)

        batch_log = self.args.batch_log

        
        all_train_losses = [] # List to store all training losses for each batch log
        train_losses, valid_losses = [], [] # List to store average training loss for each epoch

        os.makedirs(self.args.save_path, exist_ok=True)

        for epoch in range(self.args.epochs):
            self.model.train()

            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{self.args.epochs}",
            )

            curr_accumulated_loss = 0.0
            total_train_loss = 0.0

            # Training Loop
            for batch, (inputs, targets) in pbar:

                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                output = self.model.forward(inputs)
                loss = criterion(output, targets)

                loss.backward()
                optimizer.step()

                curr_accumulated_loss += loss.item()
                total_train_loss += loss.item()

                if batch > 0 and batch % batch_log == 0:
                    pbar.set_postfix(
                        batch=batch, loss=curr_accumulated_loss / batch_log
                    )

                    # log average loss for the batch and reset accumulated loss
                    all_train_losses.append(curr_accumulated_loss / batch_log)
                    curr_accumulated_loss = 0.0


            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

            # Validation Loop
            if valid_loader is not None:
                self.model.eval()
                total_val_loss = 0.0
                for batch, (inputs, targets) in enumerate(valid_loader):

                    inputs, targets = inputs.to(device), targets.to(device)

                    with torch.no_grad():
                        output = self.model.forward(inputs)
                        loss = criterion(output, targets)

                    total_val_loss += loss.item()
                
                avg_val_loss = total_val_loss / len(valid_loader)
                valid_losses.append(avg_val_loss)
                print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
                scheduler.step(avg_val_loss)
                self.model.train()
            
            # Save Model Checkpoint
            if self.args.save_every_n_epochs != 0 and (epoch + 1) % self.args.save_every_n_epochs == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.args.save_path, f"model_epoch_{epoch+1}.pt"),
                )
        
        return {
            "model": self.model,
            "all_train_losses": all_train_losses,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
        }


@dataclass
class TrainingArguments:

    lr: float = 1e-4
    weight_decay: float = 1e-5
    optimizer_class: type = optim.AdamW
    optimizer_kwargs: dict = None
    criterion: nn.Module = nn.CrossEntropyLoss(ignore_index=-1)
    scheduler: optim.lr_scheduler._LRScheduler = optim.lr_scheduler.ReduceLROnPlateau
    epochs: int = 1
    batch_log: int = 100
    device: str = None
    save_path: str = "checkpoints/"
    save_every_n_epochs: int = 1 # set to 0 to disable saving checkpoints
