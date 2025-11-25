# Code/model_b/train_b.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .cnn_model import SimpleCNN
from .data_loader_b import get_dataloaders
from .metrics_b import evaluate_metrics
from .plot_b import plot_learning_curve


def train_model_b(
    batch_size=64,
    lr=1e-3,
    epochs=1,
    augment=True,
    subset_ratio=1.0,      # NEW: training budget
    model_size="small",     # NEW: model capacity
    device=None
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Model B - CNN] Training on device: {device}")
    print(f"  → augment={augment}, subset_ratio={subset_ratio}, model_size={model_size}")

    # ----- Load dataset -----
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        augment_train=augment
    )

    # If subset training is requested
    # ------- Training Budget -------
    if subset_ratio < 1.0:
        new_len = int(len(train_loader.dataset) * subset_ratio)
        print(f"  → Using {new_len}/{len(train_loader.dataset)} samples (training budget)")

        subset_ds = torch.utils.data.Subset(train_loader.dataset, range(new_len))

        # rebuild loader
        train_loader = torch.utils.data.DataLoader(
            subset_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

    # ----- Define different capacity models -----
    if model_size == "small":
        model = SimpleCNN(num_classes=2, channels=[8, 16, 32]).to(device)
    elif model_size == "medium":
        model = SimpleCNN(num_classes=2, channels=[16, 32, 64]).to(device)
    elif model_size == "large":
        model = SimpleCNN(num_classes=2, channels=[32, 64, 128]).to(device)
    else:
        raise ValueError("Unknown model_size")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, val_accuracies = [], [], []

    # ===== Training Loop =====
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

        # ----- Validation -----
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

    # ----- Save learning curves -----
    plot_learning_curve(train_losses, val_losses, val_accuracies, suffix=model_size)

    # ----- Final Test Evaluation -----
    print("\n=== Test Evaluation ===")
    results = evaluate_metrics(model, test_loader, device)
    print(results)

    # ----- Save model -----
    torch.save(model.state_dict(), f"model_b/cnn_{model_size}.pth")

    return results