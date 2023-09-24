import torch
from collections import Counter
from sklearn.metrics import confusion_matrix

from NN10b.netrual_network_structure import retrain_with_top_features


def get_confusion_matrix(targets, predictions, num_classes=3):
    return confusion_matrix(targets, predictions, labels=range(num_classes))

def calculate_metrics(targets, predictions, num_classes=3):
    cm = get_confusion_matrix(targets, predictions, num_classes)
    precision = list()
    recall = list()
    f1 = list()
    specificity = list()

    for i in range(num_classes):
        tp = cm[i, i]
        fp = sum(cm[j, i] for j in range(num_classes)) - tp
        fn = sum(cm[i, j] for j in range(num_classes)) - tp
        tn = sum(sum(cm[j, k] for k in range(num_classes)) for j in range(num_classes)) - tp - fp - fn

        precision.append(tp / (tp + fp) if (tp + fp) != 0 else 0)
        recall.append(tp / (tp + fn) if (tp + fn) != 0 else 0)
        specificity.append(tn / (tn + fp) if (tn + fp) != 0 else 0)
        f1.append(2 * precision[-1] * recall[-1] / (precision[-1] + recall[-1]) if (precision[-1] + recall[-1]) != 0 else 0)

    precision = sum(precision) / num_classes
    recall = sum(recall) / num_classes
    f1 = sum(f1) / num_classes
    specificity = sum(specificity) / num_classes
    gmean = (recall * specificity) ** 0.5

    return precision, recall, f1, specificity, gmean

def evaluate_model(net, X_test_tensor, Y_test_tensor):
    """
    Evaluate a trained model on test data.

    Args:
    - net: The trained neural network model.
    - X_test_tensor: Test data features.
    - Y_test_tensor: True labels for the test data.

    Returns:
    A dictionary containing various evaluation metrics.
    """
    with torch.no_grad():
        test_outputs = net(X_test_tensor)
        _, predicted = torch.max(test_outputs, 1)
        correct = (predicted == Y_test_tensor).sum().item()
        accuracy = correct / Y_test_tensor.size(0)

        # Calculate metrics
        precision, recall, f1, specificity, gmean = calculate_metrics(Y_test_tensor, predicted)

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Specificity": specificity,
            "Geometric Mean": gmean
        }

    return metrics


def train_evaluate_average(feature_method, method_name, X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, col_dict,
                   n_trials=10):
    metrics_accumulator = {
        "Accuracy": 0,
        "Precision": 0,
        "Recall": 0,
        "F1 Score": 0,
        "Specificity": 0,
        "Geometric Mean": 0
    }

    # Counter to track feature occurrences over iterations
    feature_counter = Counter()

    for _ in range(n_trials):
        top_12_indices = feature_method(X_train_tensor.numpy(), Y_train_tensor.numpy(),12)
        for index in top_12_indices:
            feature_counter[index] += 1

        net = retrain_with_top_features(top_12_indices, X_train_tensor, Y_train_tensor)
        X_test_selected = X_test_tensor[:, top_12_indices]
        metrics = evaluate_model(net, X_test_selected, Y_test_tensor)

        for key, value in metrics.items():
            metrics_accumulator[key] += value

    # Calculate average metrics
    for key in metrics_accumulator:
        metrics_accumulator[key] /= n_trials

    # Get the most common 12 features over the iterations
    most_common_indices = [item[0] for item in feature_counter.most_common(12)]
    most_common_columns = [col_dict[i] for i in most_common_indices]


    print(f"{method_name} most common top 12 columns over {n_trials} trials:", most_common_columns)
    print(f"use {method_name} method: the average metric result over {n_trials} trials:")
    for key, value in metrics_accumulator.items():
        print(f"{key}: {value:.2f}")
    print("\n")