"""
Inference Script
Evaluate trained models on test sets
"""

import argparse

from src.utils.data_loader import load_test_dataset

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model weights')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist', help='Dataset to evaluate on')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[128, 64], help='List of hidden layer sizes')
    parser.add_argument('--activation', type=str, choices=['relu', 'sigmoid', 'tanh'], default='relu', help='Activation function to use')

    
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    pass



def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """

    pass




def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    print("Loading model...")
    model = load_model(args.model_path)
    print("Loading test dataset...")
    X_test, y_test = load_test_dataset(args.dataset)
    print("Evaluating model...")
    results = evaluate_model(model, X_test, y_test)
    for key, value in results.items():
        print(f"{key}: {value}")
        
    
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
