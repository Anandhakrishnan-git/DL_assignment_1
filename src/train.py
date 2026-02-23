"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist', help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], default='adam', help='Optimizer to use')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[128, 64], help='List of hidden layer sizes')
    parser.add_argument('--activation', type=str, choices=['relu', 'sigmoid', 'tanh'], default='relu', help='Activation function to use')
    parser.add_argument('--loss', type=str, choices=['cross_entropy', 'mse'], default='cross_entropy', help='Loss function to use')
    parser.add_argument('--weight_init', type=str, choices=['random', 'xavier', 'he'], default='xavier', help='Weight initialization method')
    parser.add_argument('--wandb_project', type=str, default='nn_training', help='W&B project name for logging')
    parser.add_argument('--model_save_path', type=str, default='models/', help='Path to save trained model (do not give absolute path, rather provide relative path)')

    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()
    
    
    print("Training complete!")


if __name__ == '__main__':
    main()
