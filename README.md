# InMindCNN

Train and evaluate a simple CNN on CIFAR-10 using PyTorch.

## Quickstart

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd inmindCNN
   ```

2. **Install the uv Python package manager (faster than pip):**
   ```bash
   pip install uv
   ```

3. **Install all dependencies defined in `pyproject.toml`:**
   ```bash
   uv sync
   ```

2. **Edit `config.yaml`** for hyperparameters and paths if needed.
   - `val_split` controls the fraction of training data used for validation (default: 0.1).

3. **Run training:**
    Run inside uv venv:
    ```bash
    uv run train.py
    ```
    - Uses GPU if available
    - CIFAR-10 is auto-downloaded to `data/cifar10/`
    - Model weights saved to `weights/checkpoint.pth` by default.
    - After each epoch, validation loss and accuracy are reported.

## Validation
- A portion of the training set is used for validation (see `val_split` in config).
- After each epoch, validation loss and accuracy are printed.
- Final test loss and accuracy are printed after training.