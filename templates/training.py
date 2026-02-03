
from template import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, valid_loader=None, epochs=10, lr=1e-4,
                criterion=None, optimizer=None, scheduler=None,
                device=None, batch_log=100):
    """
    Trains a PyTorch model on provided datasets with optional validation.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        valid_loader (DataLoader, optional): DataLoader for the validation dataset.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate for the optimizer.
        criterion (nn.Module, optional): Loss function. Defaults to CrossEntropyLoss.
        optimizer (torch.optim.Optimizer, optional): Optimizer. Defaults to Adam.
        scheduler (torch.optim.lr_scheduler, optional): LR scheduler.
            Defaults to ReduceLROnPlateau with patience=3.
        device (str or torch.device, optional): Device to train on ('cuda' or 'cpu').
            Automatically selects GPU if available.
        batch_log (int): Interval (number of batches) to log training loss.

    Returns:
        dict: Dictionary containing the trained model and lists of training and validation losses.
    """

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optimizer or optim.Adam(model.parameters(), lr=lr)
    criterion = criterion or nn.CrossEntropyLoss(ignore_index=-1)
    scheduler = scheduler or optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    train_losses, valid_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{epochs}")

        # Training loop
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if (batch_idx + 1) % batch_log == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss = epoch_loss / (batch_idx + 1)
                pbar.set_postfix({'Training Loss': f"{avg_loss:.4f}"})

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1} Training Loss: {avg_epoch_loss:.4f}")

        # Validation loop
        if valid_loader:
            model.eval()
            valid_loss = 0.0

            with torch.no_grad():
                for inputs_val, targets_val in valid_loader:
                    inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
                    outputs_val = model(inputs_val)
                    loss_val = criterion(outputs_val, targets_val)
                    valid_loss += loss_val.item()

            avg_valid_loss = valid_loss / len(valid_loader)
            valid_losses.append(avg_valid_loss)
            print(f"Epoch {epoch+1} Validation Loss: {avg_valid_loss:.4f}")

            # Step scheduler based on validation loss if applicable
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_valid_loss)
            else:
                scheduler.step()

    return {
        'model': model,
        'train_losses': train_losses,
        'valid_losses': valid_losses
    }
