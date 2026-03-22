# DL Assignment 1: NumPy MLP for MNIST / Fashion-MNIST

This repository contains a from-scratch Multi-Layer Perceptron (MLP) implementation using NumPy for image classification on `mnist` and `fashion_mnist`.

## What Is Included

- Dense neural network implementation (`src/ann/`)
- Forward and backward propagation from scratch
- Activations: `relu`, `sigmoid`, `tanh` (+ `softmax` at output for loss)
- Losses: `cross_entropy`, `mse`
- Optimizers: `sgd`, `momentum`, `nag`, `rmsprop`
- Training script with validation/test evaluation and best-model tracking
- Inference script for loading saved weights and computing metrics
- Analysis notebooks for assignment questions (`notebooks/q2_*.ipynb`)

## Repository Structure

- `src/train.py`: Train model, evaluate, save model + best config
- `src/inference.py`: Load saved model and run test-set evaluation
- `src/ann/`: Neural net, layers, activations, optimizers, losses
- `src/utils/data_loader.py`: Keras dataset loading + preprocessing
- `src/best_config.json`: Best run metadata (updated by training)
- `src/best_model.npy`: Best model weights (updated by training)
- `notebooks/`: Assignment analysis notebooks

## Setup

1. Create/activate a Python environment (recommended).
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Training

Run from repository root:

```bash
python src/train.py
```

Example with custom hyperparameters:

```bash
python src/train.py -d mnist -e 20 -b 64 -lr 0.01 -wd 0.0001 -o momentum -nhl 2 -sz 128 128 -a relu -l cross_entropy -w_i xavier --model_save_path models/
```

Key CLI options:

- `-d, --dataset`: `mnist | fashion_mnist`
- `-o, --optimizer`: `sgd | momentum | nag | rmsprop`
- `-a, --activation`: `relu | sigmoid | tanh`
- `-l, --loss`: `cross_entropy | mse`
- `-w_i, --weight_init`: `random | xavier | zeros`
- `-sz, --hidden_size`: one or more hidden layer sizes (space-separated)

## Inference

Evaluate a saved model on test data:

```bash
python src/inference.py --model_path src/best_model.npy -d mnist -b 64
```

Outputs include:

- `loss`
- `accuracy`
- `precision` (macro)
- `recall` (macro)
- `f1` (macro)
- `logits` shape

## Data Processing

`src/utils/data_loader.py`:

- Loads data using `keras.datasets`
- Flattens images to `784`-D vectors
- Normalizes pixel values to `[0, 1]`
- Creates one-hot labels
- Splits training into train/validation (default `val_ratio=0.1`)

## Notebooks

`notebooks/` contains Q2 analysis notebooks (exploration, sweeps, optimizer comparison, error analysis, etc.).

## Links to wanb and github

`Github` : https://github.com/Anandhakrishnan-git/DL_assignment_1

`wanb` : [https://wandb.ai/anandhakrishnanm21-indian-institute-of-technology-madras/q2_5_dead_neurons/reports/Multi-Layer-Perceptron-for-Image-Classification--VmlldzoxNjExNjEzMA
](https://api.wandb.ai/links/anandhakrishnanm21-indian-institute-of-technology-madras/cimlcdav)
## License

This project is licensed under the terms in [LICENSE](LICENSE).
