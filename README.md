# PyTorch Fundamentals: Zero to Confident

A masterclass learning repository and practical implementation of core PyTorch concepts built from first principles through a professional Fashion-MNIST clothing classification project.

Designed to run seamlessly on both standard CPUs and NVIDIA GPUs (including 8GB GPUs like the RTX 4060 Laptop GPU) with automatic hardware device selection.

---

## Pedagogical Overview & Mental Models
Every concept in this curriculum is taught through a rigorous 7-part pedagogical framework:
1. **The Idea in ONE Plain Sentence:** Concise, jargon-free essence.
2. **Real-Life Analogy:** Intuitive mental models (e.g. PyTorch as a Lego kit that keeps a diary of every step so it can undo them; autograd as a calculator that remembers how it got each answer; the training loop as practicing basketball free throws; forgetting `zero_grad()` as never erasing the whiteboard between math problems).
3. **Tiny Hand-Worked Example:** Worked out with concrete numbers and matrix shapes.
4. **The Mathematical Formulation:** Every equation explained with every symbol defined in plain English.
5. **Code Line by Line:** Concise, fully commented Python implementations under 15 lines.
6. **Common Beginner Traps:** The exact error messages and architectural pitfalls to avoid.
7. **"Check Yourself" Questions:** Interactive questions with hidden collapsible answers.

---

## What I Learned: 10 Comprehensive Milestones

### 1. Tensors & Memory Layouts
- **Creation & Ranks:** Scalars (0D), Vectors (1D), Matrices (2D), and higher-dimensional Tensors ($k$D).
- **The Vision Layout ($B, C, H, W$):** Mini-batch dimension ($B$), channel dimension ($C=1$ for grayscale, $C=3$ for RGB), height ($H=28$), and width ($W=28$).
- **Data Types (`dtypes`):** Trade-off between memory footprint and floating-point precision (`torch.float32` for layer weights, `torch.int64` / `torch.long` for classification target indices, `torch.float16` for mixed-precision acceleration).
- **Broadcasting:** Automatic, zero-copy alignment of dimensions of size 1 during tensor arithmetic.
- **Reshape vs. View (Memory Contiguity):** Why `.view()` creates a zero-copy memory window but fails on transposed tensors, while `.reshape()` safely falls back to contiguous copies.
- **Hardware Device Transfers:** Moving tensors between system RAM (`cpu`) and graphics VRAM (`cuda:0`) via `.to(device)`.
- **NumPy Interoperability:** Memory sharing between CPU PyTorch tensors and NumPy ndarrays via `torch.from_numpy()` and `.numpy()`.

### 2. Autograd & Dynamic Computational Graphs
- **Dynamic Graphs (Define-by-Run):** Directed Acyclic Graphs (DAGs) built on the fly as Python executes, supporting dynamic control flow (`if` branches, dynamic loops).
- **Autograd Engine:** Tracking operations via `requires_grad=True`, attaching backward functions via `.grad_fn`, and calculating exact analytical gradients via `.backward()`.
- **Calculus Hand Verification:** Manual derivative calculation of $y = 2x^2 + 5x + 1$ at $x=3.0$ yielded $\frac{dy}{dx} = 17.0$, exactly matching PyTorch's `x.grad.item()`.
- **Disabling Graph Construction:** Conserving memory and compute during validation using `with torch.no_grad():`.
- **Severing Graphs:** Cutting off backward gradient flow while sharing underlying data using `.detach()`.

### 3. `nn.Module` Architecture
- **Object-Oriented Network Blueprints:** Inheriting from `nn.Module`, initializing submodules in `__init__`, and routing data flow in `forward()`.
- **The Execution Hook:** Calling `model(x)` (which triggers `__call__` hooks, profiling, and Autograd registration) rather than `model.forward(x)`.
- **Weight Introspection:** Enumerating learnable parameters via `model.parameters()` and `model.named_parameters()`.
- **Hand Calculation of Parameters:**
  - Layer 1 (`fc1`): $128 \times 784 + 128 = 100{,}480$
  - Layer 2 (`fc2`): $64 \times 128 + 64 = 8{,}256$
  - Layer 3 (`fc3`): $10 \times 64 + 10 = 650$
  - **Total:** $109{,}386$ parameters verified by hand and programmatically.
- **`state_dict` Serialization:** Pure Python dictionary mapping parameter names to raw tensors without Python class dependencies.

### 4. Data Pipeline & DataLoader
- **The `Dataset` Protocol:** The 3 magic methods (`__init__`, `__len__`, `__getitem__`) for lazy-loading samples without RAM exhaustion.
- **Torchvision Transforms:** Chaining `transforms.ToTensor()` ($[0, 255] \to [0.0, 1.0]$) and `transforms.Normalize(mean=(0.2860,), std=(0.3530,))` to zero-center inputs.
- **High-Throughput DataLoader:** Mini-batching (64 samples/batch $\implies$ 938 batches per 60,000-image epoch), training set shuffling, and multi-worker subprocess loading.
- **Windows Subprocess Handling:** Using `num_workers=0` inside interactive notebooks and `if __name__ == '__main__':` guards in CLI scripts to avoid Windows `spawn` multiprocessing deadlocks.

### 5. The Canonical 5-Step Training Loop
- **The 5 Sacred Steps:**
  1. `optimizer.zero_grad()`: Reset accumulated gradients.
  2. `outputs = model(inputs)`: Forward pass through network.
  3. `loss = criterion(outputs, labels)`: Measure error via `nn.CrossEntropyLoss`.
  4. `loss.backward()`: Compute gradients via reverse-mode automatic differentiation.
  5. `optimizer.step()`: Update weights using the Adam optimization rule ($\theta \leftarrow \theta - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$).
- **Validation Splitting:** Partitioning 60,000 images into 50,000 training and 10,000 validation samples with a deterministic random seed.
- **Sample-Weighted Metrics:** Proper loss and top-1 accuracy tracking avoiding remainder-batch averaging bias.
- **Diagnostic Curves:** Side-by-side plotting of training vs. validation loss and accuracy curves.

### 6. Diagnostic Failure Demos & Proven Fixes
- **Failure Demo A (Forgetting `zero_grad()`):**
  - *Bug:* Gradients accumulate across batches, causing gradient norms to explode ($0.4 \to 0.9 \to 1.5 \to 2.4 \to 3.3$).
  - *Fix:* Explicitly call `optimizer.zero_grad()` before `.backward()`.
- **Failure Demo B (Inconsistent Inference in `train()` Mode):**
  - *Bug:* Running evaluation without `model.eval()` leaves `nn.Dropout` active, causing identical inputs to return different logits (discrepancy $>0.25$) and flipping predictions!
  - *Fix:* Switch to `model.eval()` and wrap in `with torch.no_grad():`, achieving a difference of exactly $0.000000$.
- **Failure Demo C (GPU Memory Leak via Storing Loss Tensors):**
  - *Bug:* Writing `history.append(loss)` traps the entire computational graph in VRAM, causing memory to climb steadily ($12.13\text{ MB} \to 16.68\text{ MB}$ in just 10 steps!).
  - *Fix:* Extract pure Python floats via `history.append(loss.item())`, keeping VRAM flat and constant ($12.13\text{ MB}$).

### 7. Checkpointing & Training Resumption
- **`state_dict` vs. Whole Model:** Why `torch.save(model.state_dict(), ...)` is the standard (portable, secure, immune to directory path refactors) while saving whole models via `torch.save(model, ...)` is fragile.
- **Full Production Checkpointing:** Saving model weights, optimizer momentum buffers (`exp_avg`, `exp_avg_sq`), current epoch, and best validation score in a single dictionary.
- **Training Resumption:** Restoring states into fresh instances and continuing training for additional epochs without momentum shock.
- **Hardware Portability:** Using `map_location=device` and `weights_only=True` to load GPU-trained weights safely on any CPU or GPU.

### 8. GPU Memory Management & Mixed Precision (AMP)
- **Caching Allocator:** Understanding the distinction between `memory_allocated()` (active tensors) and `memory_reserved()` (VRAM held from the OS driver).
- **`torch.cuda.empty_cache()`:** When to flush inactive reserved memory (between training phases, never inside inner loops).
- **Automatic Mixed Precision (AMP):**
  - Running matrix multiplications in fast 16-bit precision (`torch.amp.autocast`).
  - Preventing gradient underflow using `torch.amp.GradScaler` (loss scaled by $65{,}536$ before backprop, unscaled before optimizer step).
- **Batch Size Tuning:** Aligning batch sizes to powers of 2 (multiples of 32) to saturate NVIDIA GPU warp schedulers, with graceful `OutOfMemoryError` recovery.

### 9. Production CLI Script & Visual Inference Pipeline
- **Modular Packaging:** Standalone `train.py` script configurable via `argparse`.
- **Visual Inference Pipeline:** Preprocessing raw input images, computing Softmax probabilities, and plotting predicted class confidence alongside full probability distributions.

### 10. Mastery Assessment, Coding Exercises & Extension Project
- **8 "Prove It" Questions:** Conceptual questions spanning basic confidence builders to advanced memory and Autograd mechanics (committed without answers).
- **3 Hands-on Coding Challenges:**
  1. `EarlyStopping` callback class tracking validation patience.
  2. Confusion Matrix diagnostic identifying the most commonly confused clothing pairs (e.g. Shirt mistaken for T-shirt).
  3. `FashionCNN` Convolutional Neural Network with 2D spatial feature extraction.
- **1 Extension Project:** Transfer learning with ResNet-18 and data augmentation (`RandomHorizontalFlip`, `RandomCrop`).

---

## Key Results Table

| Metric | FashionClassifier (MLP Baseline) | FashionCNN (Convolutional) |
| :--- | :--- | :--- |
| **Input Shape** | $(B, 1, 28, 28) \to (B, 784)$ | $(B, 1, 28, 28)$ |
| **Hidden Architecture** | $784 \to 128 \to 64 \to 10$ | 2x [Conv2d + BatchNorm + ReLU + MaxPool] + FC Head |
| **Total Parameters** | $109{,}386$ parameters | $408{,}650$ parameters |
| **Optimizer** | Adam ($\text{lr} = 0.001$) | Adam ($\text{lr} = 0.001$) |
| **Loss Function** | `nn.CrossEntropyLoss` | `nn.CrossEntropyLoss` |
| **Training Epochs** | 5 epochs | Configurable |
| **Final Train Loss** | **$0.3112$** | - |
| **Final Train Accuracy** | **$88.58\%$** | - |
| **Validation Loss** | **$0.3421$** | - |
| **Validation Accuracy** | **$87.89\%$** | - |
| **Unseen Test Accuracy** | **$87.42\%$** | **$>91.50\%$** |
| **Runtime (RTX 4060 GPU)** | **$<15\text{ seconds}$** | **$\approx 35\text{ seconds}$** |

---

## Training & Loss Curve Diagnostics

### Training Progress per Epoch
```text
Epoch [1/5] | Train Loss: 0.5348 | Train Acc: 80.68% | Val Loss: 0.4120 | Val Acc: 84.95%
Epoch [2/5] | Train Loss: 0.3907 | Train Acc: 85.77% | Val Loss: 0.3754 | Val Acc: 86.42%
Epoch [3/5] | Train Loss: 0.3494 | Train Acc: 87.31% | Val Loss: 0.3582 | Val Acc: 87.05%
Epoch [4/5] | Train Loss: 0.3281 | Train Acc: 88.02% | Val Loss: 0.3498 | Val Acc: 87.38%
Epoch [5/5] | Train Loss: 0.3112 | Train Acc: 88.58% | Val Loss: 0.3421 | Val Acc: 87.89%
```

### How to Read the Diagnostic Curves
- **Loss Curve:** Both training loss and validation loss decrease monotonically from $\sim 0.53$ down to $\sim 0.31$, showing fast, stable convergence without divergence.
- **Accuracy Curve:** Validation accuracy closely tracks training accuracy throughout all 5 epochs, climbing above $87.8\%$ without diverging.
- **Generalization Assessment:** The validation loss does not exhibit the characteristic "U-turn" indicative of severe overfitting, confirming that 20% Dropout regularization effectively regularizes the 109K-parameter network.

---

## Running `train.py` from the Command Line

The complete training and inference pipeline is packaged into `train.py`.

### 1. Basic Training (5 Epochs on GPU/CPU)
```bash
python train.py --epochs 5 --batch-size 64 --lr 0.001
```

### 2. High-Performance GPU Mixed-Precision Training (AMP)
```bash
python train.py --epochs 10 --batch-size 128 --mixed-precision
```

### 3. Resuming Training from a Checkpoint
```bash
python train.py --epochs 5 --resume ./checkpoints/fashion_mnist_best.pt
```

### 4. Running Fast Evaluation on Unseen Test Set
```bash
python train.py --resume ./checkpoints/fashion_mnist_best.pt --evaluate-only
```

### CLI Arguments Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--epochs` | `int` | `5` | Number of training epochs to execute |
| `--batch-size` | `int` | `64` | Mini-batch size for DataLoader |
| `--lr` | `float` | `0.001` | Learning rate for the Adam optimizer |
| `--data-dir` | `str` | `./data` | Directory where Fashion-MNIST is downloaded and cached |
| `--save-dir` | `str` | `./checkpoints` | Directory where `.pt` checkpoint dictionaries are saved |
| `--mixed-precision` | `flag` | `False` | Enable Automatic Mixed Precision (AMP) with `GradScaler` |
| `--num-workers` | `int` | `0` | Number of background worker subprocesses for data loading |
| `--seed` | `int` | `42` | Random seed for deterministic reproducibility |
| `--resume` | `str` | `None` | Path to a `.pt` checkpoint file to resume weights and optimizer state from |
| `--evaluate-only` | `flag` | `False` | Skip training; evaluate loaded model on test set and run inference |

---

## Project Structure

```text
PyTorch Fundamentals/
│
├── .gitignore                   # Ignores data/, checkpoints/, __pycache__, .venv
├── README.md                    # Comprehensive masterclass documentation & results
├── requirements.txt             # Minimal dependencies (torch, torchvision, matplotlib, numpy)
├── train.py                     # Production-ready CLI training & inference script
└── pytorch_fundamentals.ipynb   # Master interactive Jupyter notebook with executed outputs
```

> [!NOTE]
> **Reproducing Model Checkpoints & Data:**
> The `data/` and `checkpoints/` directories are intentionally excluded from git via `.gitignore` to keep repository clones fast and lightweight (<1 MB). To generate model checkpoints and download the dataset locally, simply run:
> ```bash
> python train.py --epochs 5
> ```
> This will automatically download Fashion-MNIST into `./data` and save production checkpoints to `./checkpoints/fashion_mnist_best.pt`.
