# Code/model_b/train_b.py
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .cnn_model import AMLSCNN
from .data_loader_b import get_dataloaders
from .metrics_b import evaluate_metrics
from .plot_b import ensure_dir, plot_confusion_matrix, plot_learning_curve

def get_optimizer(optimizer_name, model_params, lr):
    """Return optimizer based on name."""
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return optim.Adam(model_params, lr=lr)

    elif optimizer_name == "sgd":
        return optim.SGD(model_params, lr=lr, momentum=0.9)

    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model_params, lr=lr, momentum=0.9)

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def train_model_b(
    batch_size=64,
    lr=1e-3,
    epochs=20,
    augment=True,
    subset_ratio=1.0,
    model_size="small",
    device="cpu",
    patience=5,          # Early Stopping
    optimizer_name="adam",
    log_dir="model_b/logs"
):
    """
    Train Model B: Simple CNN with optional data augmentation and training budget.
    arguments:
        batch_size: training batch size
        lr: learning rate
        epochs: number of training epochs
        augment: whether to use data augmentation
        subset_ratio: ratio of training data to use (for training budget experiments)
        model_size: "small", "medium", or "large" CNN
        device: "cpu" or "cuda"
        patience: epochs to wait for improvement before early stopping
        optimizer_name: "adam", "sgd", or "rmsprop"
        log_dir: directory to save logs and plots
    returns:
        results: dictionary of test results
    """
    ensure_dir(log_dir)
    log_text = []
    log_text.append(f"===== Experiment Log =====\n")
    log_text.append(f"batch_size={batch_size}\n")
    log_text.append(f"lr={lr}\n")
    log_text.append(f"epochs={epochs}\n")
    log_text.append(f"augment={augment}\n")
    log_text.append(f"subset_ratio={subset_ratio}\n")
    log_text.append(f"model_size={model_size}\n")
    log_text.append(f"device={device}\n")
    log_text.append(f"patience={patience}\n")
    log_text.append(f"optimizer_name={optimizer_name}\n\n")

    print(f"[Model B - CNN] Training on device: {device}")
    print(f"  → augment={augment}, subset_ratio={subset_ratio}, model_size={model_size}")

    # Load datasets
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        augment_train=augment
    )

    # Training Budget
    if subset_ratio < 1.0:
        new_len = int(len(train_loader.dataset) * subset_ratio)
        print(f"  → Using {new_len}/{len(train_loader.dataset)} samples (training budget)")
        log_text.append(f"Training budget used = {new_len}\n")

        subset_ds = torch.utils.data.Subset(train_loader.dataset, range(new_len))
        train_loader = torch.utils.data.DataLoader(
            subset_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

    # Model Capacity 
    if model_size == "small":
        model = AMLSCNN(num_classes=2, channels=[8, 16, 32]).to(device)
    elif model_size == "medium":
        model = AMLSCNN(num_classes=2, channels=[16, 32, 64]).to(device)
    elif model_size == "large":
        model = AMLSCNN(num_classes=2, channels=[32, 64, 128]).to(device)
    else:
        raise ValueError("Unknown model_size")

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr)

    train_losses, val_losses, val_accuracies = [], [], []

    # Early Stopping 
    best_val_loss = float("inf")
    patience_counter = 0

    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        total, correct = 0, 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item()

                preds = outputs.argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_acc = correct / total
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        log_text.append(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}\n")

        # Early Stopping 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            print(f"  → EarlyStopping counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered! Stopping training early.")
            model.load_state_dict(best_model_state)
            break

    # Save learning curves 
    plot_learning_curve(train_losses, val_losses, val_accuracies, suffix=model_size, log_dir=log_dir)

    # Final Test Evaluation
    print("\n=== Test Evaluation ===")
    results, preds, labels = evaluate_metrics(model, test_loader, device, return_preds=True)
    print(results)
    log_text.append("\n=== Test Evaluation ===\n")
    for metric, value in results.items():
        log_text.append(f"{metric.capitalize()}: {value:.4f}\n")

    cm = confusion_matrix(labels, preds)
    suffix = f"{model_size}_aug{augment}_budget{subset_ratio}"
    plot_confusion_matrix(cm, suffix=suffix, log_dir=log_dir)

    ensure_dir("model_b/saved_models")
    torch.save(model.state_dict(), f"model_b/saved_models/cnn_{model_size}_aug{augment}_budget{subset_ratio}_optim{optimizer_name}.pth")

    return results
