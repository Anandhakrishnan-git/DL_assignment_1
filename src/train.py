"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import json
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist', help='Dataset to train on')
    parser.add_argument('-e', '--epochs', type=int, default= 40, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default= 64, help='Mini-batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default= 0.034369755404477266, help='Learning rate for optimizer')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.00002152989422140758, help='Weight decay for L2 regularization')
    parser.add_argument('-o', '--optimizer', type=str, choices=['sgd', 'momentum', 'nag', 'rmsprop'], default='momentum', help='Optimizer to use')
    parser.add_argument('-nhl', '--num_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128,128], help='Number of neurons in each hidden layer')
    parser.add_argument('-a', '--activation', type=str, choices=['relu', 'sigmoid', 'tanh'], default='relu', help='Activation function to use')
    parser.add_argument('-l', '--loss', type=str, choices=['cross_entropy', 'mse'], default='cross_entropy', help='Loss function to use')
    parser.add_argument('-w_i', '--weight_init', type=str, choices=['random', 'xavier', 'zeros'], default='xavier', help='Weight initialization method')
    parser.add_argument('-w_p', '--wandb_project', type=str, default='nn_training', help='W&B project name for logging')
    parser.add_argument('--model_save_path', type=str, default='src/', help='Path to save trained model (relative path)')

    return parser.parse_args()



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
    history = model.train(
        X_train,
        Y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_val=X_val,
        y_val=Y_val,
    )

    # print("Evaluating on validation and test sets...")
    # val_loss, val_acc = model.evaluate(X_val, Y_val)
    # test_loss, test_acc = model.evaluate(X_test, Y_test)
    # test_logits = model.forward(X_test)
    # y_true = np.argmax(Y_test, axis=1)
    # y_pred = np.argmax(test_logits, axis=1)
    # test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
    #     y_true, y_pred, average='macro', zero_division=0
    # )



    # src_dir = os.path.dirname(__file__)
    # best_model_path = os.path.join(src_dir, "best_model.npy")
    # best_config_path = os.path.join(src_dir, "best_config.json")

    # best_so_far = -1.0
    # if os.path.exists(best_config_path):
    #     with open(best_config_path, "r", encoding="utf-8") as f:
    #         prev = json.load(f)
    #         best_so_far = float(prev.get("test_f1", -1.0))

    # if test_f1 >= best_so_far:
    #     np.save(best_model_path, model.get_weights(), allow_pickle=True)
    #     best_config = dict(vars(args))
    #     best_config.update(
    #         {
    #             "val_loss": float(val_loss),
    #             "val_acc": float(val_acc),
    #             "test_loss": float(test_loss),
    #             "test_acc": float(test_acc),
    #             "test_precision": float(test_precision),
    #             "test_recall": float(test_recall),
    #             "test_f1": float(test_f1),
    #         }
    #     )
    #     with open(best_config_path, "w", encoding="utf-8") as f:
    #         json.dump(best_config, f, indent=2)

    # print(f"Validation | loss={val_loss:.4f}, accuracy={val_acc:.4f}")
    # print(f"Test       | loss={test_loss:.4f}, accuracy={test_acc:.4f}")
    # print(f"Test       | precision={test_precision:.4f}, recall={test_recall:.4f}, f1={test_f1:.4f}")
    # print(f"Best model path: {best_model_path}")
    # print(f"Best config path: {best_config_path}")
    print("Training complete!")


if __name__ == '__main__':
    main()
