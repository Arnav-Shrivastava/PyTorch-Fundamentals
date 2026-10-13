#!/usr/bin/env python3
"""
PyTorch Fundamentals: Fashion-MNIST Production Training & Inference Pipeline
Author: Senior ML Mentor for Arnav
Description:
    A modular, production-ready PyTorch training and evaluation script with:
    - Automatic hardware device selection (CUDA GPU or CPU)
    - Full reproducibility seeding
    - torchvision dataset downloads and transforms
    - Modular 5-step training loop with accuracy and loss tracking
    - Optional Automatic Mixed Precision (AMP) with GradScaler
    - Comprehensive checkpointing (model weights + optimizer momentum)
    - Single-image inference function with softmax confidence scores
    - CLI arguments via argparse
"""

import os
import time
import argparse
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms


# ==============================================================================
# 1. NEURAL NETWORK ARCHITECTURE
# ==============================================================================
class FashionClassifier(nn.Module):
    """
    Multilayer Perceptron (MLP) for Fashion-MNIST classification.
    Input: (B, 1, 28, 28) grayscale images -> flattened to (B, 784).
    Hidden: 784 -> 128 (ReLU) -> 64 (ReLU) -> Dropout(0.2).
    Output: 64 -> 10 raw class logits.
    Total parameters: 109,386.
    """
    def __init__(self, in_features: int = 784, num_classes: int = 10, dropout_p: float = 0.2):
        super().__init__()
        # First fully connected layer: 784 inputs -> 128 hidden units
        self.fc1 = nn.Linear(in_features, 128)
        # Non-linear activation function
        self.relu = nn.ReLU()
        # Second fully connected layer: 128 hidden units -> 64 hidden units
        self.fc2 = nn.Linear(128, 64)
        # Dropout regularization layer to prevent neuron co-adaptation
        self.dropout = nn.Dropout(p=dropout_p)
        # Final classification head: 64 hidden units -> 10 class logits
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input spatial dimensions if 4D: (B, C, H, W) -> (B, 784)
        x = x.reshape(x.shape[0], -1)
        # Pass through first layer with ReLU activation
        x = self.relu(self.fc1(x))
        # Pass through second layer with ReLU activation
        x = self.relu(self.fc2(x))
        # Apply dropout (active only during training)
        x = self.dropout(x)
        # Output raw logits (no softmax here; nn.CrossEntropyLoss handles it)
        logits = self.fc3(x)
        return logits


# ==============================================================================
# 2. DATA PIPELINE
# ==============================================================================
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def get_data_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_split: int = 10000,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Downloads Fashion-MNIST and builds train, validation, and test DataLoaders.
    Applies standard ToTensor and empirical channel normalization (mean=0.2860, std=0.3530).
    """
    # Compose data transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860,), std=(0.3530,))
    ])

    # Download full training set (60,000 samples)
    full_train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    # Download test set (10,000 samples)
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Partition training data into train and validation subsets
    train_size = len(full_train_dataset) - val_split
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_train_dataset, [train_size, val_split], generator=generator
    )

    # Build DataLoaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


# ==============================================================================
# 3. TRAINING & EVALUATION FUNCTIONS
# ==============================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    use_amp: bool = False
) -> Tuple[float, float]:
    """
    Executes one full training epoch using the explicit 5-step optimization loop.
    Supports optional Automatic Mixed Precision (AMP).
    """
    model.train()
    total_loss, correct_count, total_samples = 0.0, 0, 0
    device_type = "cuda" if device.type == "cuda" else "cpu"

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # STEP 1: Clear old gradients
        optimizer.zero_grad()

        # STEP 2 & 3: Forward pass and Loss computation (with optional AMP)
        if use_amp and device.type == "cuda" and scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            # STEP 4: Backward pass with scaled gradients
            scaler.scale(loss).backward()
            # STEP 5: Unscale and step optimizer
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            # STEP 4: Standard backward pass
            loss.backward()
            # STEP 5: Standard optimizer step
            optimizer.step()

        # Accumulate metrics using .item() to prevent GPU memory leaks
        batch_size = len(labels)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct_count += (preds == labels).sum().item()
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_acc = (correct_count / total_samples) * 100.0
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluates model on validation or test DataLoader.
    Disables dropout via model.eval() and frees graph tracking via torch.no_grad().
    """
    model.eval()
    total_loss, correct_count, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = len(labels)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct_count += (preds == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    acc_percent = (correct_count / total_samples) * 100.0
    return avg_loss, acc_percent


# ==============================================================================
# 4. CHECKPOINTING & INFERENCE
# ==============================================================================
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    filepath: str
) -> None:
    """Saves a complete production checkpoint dictionary to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_accuracy": val_acc
    }
    torch.save(checkpoint, filepath)
    print(f"-> Checkpoint saved to: {filepath}")


def predict_image(
    image_tensor: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    class_names: list = CLASS_NAMES
) -> Dict[str, Any]:
    """
    Performs inference on a single image tensor.
    Returns predicted class, confidence percentage, and top probabilities.
    """
    model.eval()
    # Ensure tensor has batch dimension: (1, 1, 28, 28)
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    elif image_tensor.ndim == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        # Convert raw logits to probabilities summing to 1.0 using softmax
        probs = torch.softmax(logits, dim=1).squeeze()
        top_prob, top_idx = torch.topk(probs, k=1)

    predicted_class = class_names[top_idx.item()]
    confidence_pct = top_prob.item() * 100.0

    return {
        "predicted_class": predicted_class,
        "confidence_pct": confidence_pct,
        "all_probabilities": probs.cpu().numpy()
    }


# ==============================================================================
# 5. MAIN CLI RUNNER
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or evaluate Fashion-MNIST classifier the professional PyTorch way."
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for Fashion-MNIST data")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Directory for saved checkpoints")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable Automatic Mixed Precision (AMP)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes (default: 0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--evaluate-only", action="store_true", help="Run evaluation on test set and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Device selection
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("=" * 60)
    print("PYTORCH FASHION-MNIST TRAINING & INFERENCE PIPELINE")
    print("=" * 60)
    print(f"Device:           {device}")
    if use_cuda:
        print(f"GPU Hardware:     {torch.cuda.get_device_name(0)}")
    print(f"Mixed Precision:  {args.mixed_precision and use_cuda}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Learning Rate:    {args.lr}")
    print(f"Total Epochs:     {args.epochs}")
    print("=" * 60)

    # 2. Build DataLoaders
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )

    # 3. Instantiate model, loss criterion, and optimizer
    model = FashionClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4. Optional AMP GradScaler
    scaler = None
    if args.mixed_precision and use_cuda:
        scaler = torch.amp.GradScaler("cuda", enabled=True)

    start_epoch = 1

    # 5. Handle checkpoint resumption
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading checkpoint from: {args.resume}")
            chk = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(chk["model_state_dict"])
            if "optimizer_state_dict" in chk and not args.evaluate_only:
                optimizer.load_state_dict(chk["optimizer_state_dict"])
            start_epoch = chk.get("epoch", 0) + 1
            print(f"Successfully resumed model state. Resuming from epoch {start_epoch}.")
        else:
            print(f"Warning: Checkpoint path '{args.resume}' does not exist! Starting from scratch.")

    # 6. Evaluation-only mode
    if args.evaluate_only:
        print("\nRunning Evaluation Mode...")
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
        return

    # 7. Training loop
    best_val_acc = 0.0
    t0 = time.time()
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.mixed_precision
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch}/{start_epoch + args.epochs - 1}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save checkpoint if best validation accuracy achieved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.save_dir, "fashion_mnist_best.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_path)

    print(f"\nTraining completed in {time.time() - t0:.2f} seconds!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # 8. Final evaluation on unseen test dataset
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    print("=" * 60)
    print(f"FINAL UNSEEN TEST EVALUATION:")
    print(f"Test Loss:     {final_test_loss:.4f}")
    print(f"Test Accuracy: {final_test_acc:.2f}%")
    print("=" * 60)

    # 9. Demonstration inference on sample test image
    sample_img, sample_lbl = test_loader.dataset[0]
    result = predict_image(sample_img, model, device)
    print(f"Sample Inference Test:")
    print(f"Ground Truth Label: {CLASS_NAMES[sample_lbl]}")
    print(f"Predicted Class:    {result['predicted_class']}")
    print(f"Confidence:         {result['confidence_pct']:.2f}%")


if __name__ == "__main__":
    main()
