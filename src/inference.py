"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

try:
    from ann.neural_network import NeuralNetwork
    from utils.data_loader import load_dataset
except ModuleNotFoundError:
    # Support package-style imports when used as `import src.inference`.
    from .ann.neural_network import NeuralNetwork
    from .utils.data_loader import load_dataset


def parse_arguments():
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'fashion_mnist'],
                        default='mnist', help='Dataset to evaluate on')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='Number of training epochs (kept for CLI parity)')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('-lr', '--learning_rate', type=float, default= 0.034369755404477266, help='Learning rate (kept for CLI parity)')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.00002152989422140758, help='Weight decay (kept for CLI parity)')
    parser.add_argument('-o', '--optimizer', type=str, choices=['sgd', 'momentum', 'nag', 'rmsprop'],
                        default='momentum', help='Optimizer (kept for CLI parity)')
    parser.add_argument('-nhl', '--num_layers', type=int, default=2,
                        help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 128],
                        help='Number of neurons in each hidden layer')
    parser.add_argument('-a', '--activation', type=str, choices=['sigmoid', 'tanh', 'relu'],
                        default='relu', help='Activation function for hidden layers')
    parser.add_argument('-l', '--loss', type=str, choices=['cross_entropy', 'mse'],
                        default='cross_entropy', help='Loss function (kept for CLI parity)')
    parser.add_argument('-w_i', '--weight_init', type=str, choices=['random', 'xavier', 'zeros'],
                        default='xavier', help='Weight initialization method')
    parser.add_argument('-w_p', '--wandb_project', type=str, default='nn_training', help='W&B project name (kept for CLI parity)')
    parser.add_argument('--model_save_path', type=str, default='models/', help='Model save path (kept for CLI parity)')
    parser.add_argument('--model_path', type=str,default= 'src/best_model.npy' ,
                        help='Path to saved model weights (relative path)')

    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    model_data = np.load(model_path, allow_pickle=True).item()

    # Handle both saved formats:
    # 1) metadata dict with "weights"/"biases"
    # 2) plain weight dict with W0, b0, ...
    if "weights" in model_data and "biases" in model_data:
        args = argparse.Namespace(
                num_layers=len(model_data["hidden_sizes"]),
                hidden_size=model_data["hidden_sizes"],
                activation=model_data["activation_name"],
                weight_init=model_data["weight_init"],
                loss=model_data["loss_name"],
                learning_rate=model_data["learning_rate"],
                weight_decay=model_data.get("weight_decay", 0.0),
                optimizer=model_data["optimizer"],
            )

        model = NeuralNetwork(args)
        for i, layer in enumerate(model.layers):
            layer.W = model_data["weights"][i]
            layer.b = model_data["biases"][i]
        return model

    layer_ids = sorted(
        int(k[1:]) for k in model_data.keys() if k.startswith("W") and k[1:].isdigit()
    )
    hidden_sizes = [int(model_data[f"W{i}"].shape[1]) for i in layer_ids[:-1]]
    args = argparse.Namespace(
        num_layers=len(hidden_sizes),
        hidden_size=hidden_sizes,
        activation="relu",
        weight_init="xavier",
        loss="cross_entropy",
        learning_rate=0.001,
        weight_decay=0.0,
        optimizer="sgd",
    )
    model = NeuralNetwork(args)
    model.set_weights(model_data)
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
    _,_,_,_, X_test, y_test = load_dataset(name=args.dataset)
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
