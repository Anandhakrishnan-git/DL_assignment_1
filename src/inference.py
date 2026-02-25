"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_test_dataset


def parse_arguments():
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'fashion_mnist'],
                        default='mnist', help='Dataset to evaluate on')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('-nhl', '--num_layers', type=int, default=2,
                        help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 64],
                        help='Number of neurons in each hidden layer')
    parser.add_argument('-a', '--activation', type=str, choices=['sigmoid', 'tanh', 'relu'],
                        default='relu', help='Activation function for hidden layers')
    parser.add_argument('-w', '--weight_init', type=str, choices=['random', 'xavier'],
                        default='xavier', help='Weight initialization method')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model weights (relative path)')

    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    model_data = np.load(model_path, allow_pickle=True).item()
        
    args = argparse.Namespace(
            num_layers=len(model_data["hidden_sizes"]),
            hidden_size=model_data["hidden_sizes"],
            activation=model_data["activation_name"],
            weight_init=model_data["weight_init"],
            loss=model_data["loss_name"],
            learning_rate=model_data["learning_rate"],
            optimizer=model_data["optimizer"],
        )
 
    model = NeuralNetwork(args) 
    for i, layer in enumerate(model.layers):
        layer.W = model_data["weights"][i]
        layer.b = model_data["biases"][i]
    return model


def evaluate_model(model, X_test, y_test, batch_size=64):
    """
    Evaluate model on test data.

    Returns:
        Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    logits = []
    for start in range(0, X_test.shape[0], batch_size):
        end = start + batch_size
        batch_logits = model.forward(X_test[start:end])
        logits.append(batch_logits)
    logits = np.vstack(logits)

    loss, accuracy = model.evaluate(X_test, y_test)

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(logits, axis=1) 
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    return {
        "logits": logits,
        "loss": float(loss),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def main():
    """
    Main inference function.
    """
    args = parse_arguments()
    print("Loading model...")
    model = load_model(args.model_path)
    print("Loading test dataset...")
    X_test, y_test = load_test_dataset(args.dataset)
    print("Evaluating model... \n") 

    results = evaluate_model(model, X_test, y_test, batch_size=args.batch_size)

    for key, value in results.items():
        if key == "logits":
            print(f"{key}: shape={value.shape}")
        else:
            print(f"{key}: {value}")
    print("\nEvaluation complete!")

    return results


if __name__ == '__main__':
    main()
