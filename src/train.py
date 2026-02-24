"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import os

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist', help='Dataset to train on')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Mini-batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('-o', '--optimizer', type=str, choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], default='sgd', help='Optimizer to use')
    parser.add_argument('-nhl', '--num_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 64], help='Number of neurons in each hidden layer')
    parser.add_argument('-a', '--activation', type=str, choices=['relu', 'sigmoid', 'tanh'], default='relu', help='Activation function to use')
    parser.add_argument('-l', '--loss', type=str, choices=['cross_entropy', 'mse'], default='cross_entropy', help='Loss function to use')
    parser.add_argument('-w_i', '--weight_init', type=str, choices=['random', 'xavier', 'he'], default='xavier', help='Weight initialization method')
    parser.add_argument('--wandb_project', type=str, default='nn_training', help='W&B project name for logging')
    parser.add_argument('--model_save_path', type=str, default='models/', help='Path to save trained model (relative path)')

    return parser.parse_args()


def resolve_model_path(path, dataset):
    if path.endswith(".npz"):
        return path
    if path.endswith("/") or path.endswith("\\") or os.path.isdir(path):
        return os.path.join(path, f"{dataset}_mlp.npy")
    return f"{path}.npy"


def main():
    """
    Main training function.
    """
    args = parse_arguments()

    print("Loading dataset...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_dataset(args.dataset)

    print("Initializing model...")
    model = NeuralNetwork(args)

    print("Training...")
    model.train(
        X_train,
        Y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_val=X_val,
        y_val=Y_val,
    )

    val_loss, val_acc = model.evaluate(X_val, Y_val)
    test_loss, test_acc = model.evaluate(X_test, Y_test)

    model_path = resolve_model_path(args.model_save_path, args.dataset)
    model.save_model(model_path)

    print(f"Validation | loss={val_loss:.4f}, accuracy={val_acc:.4f}")
    print(f"Test       | loss={test_loss:.4f}, accuracy={test_acc:.4f}")
    print(f"Model saved to: {model_path}")
    print("Training complete!")


if __name__ == '__main__':
    main()
