import torch
from collections import Counter
from NN10b.data_preprocessing import load_processed_data
from NN10b.metric import evaluate_model
from NN10b.netrual_network_structure import NeuralNetwork, train_network, retrain_with_top_features
from NN10b.feature_select_method import calculate_importance

if __name__ == "__main__":
    N_TRIALS = 10
    metrics_accumulator = {
        "Accuracy": 0,
        "Precision": 0,
        "Recall": 0,
        "F1 Score": 0,
        "Specificity": 0,
        "Geometric Mean": 0
    }

    feature_counter = Counter()

    for _ in range(N_TRIALS):
        net = NeuralNetwork(input_size=25)
        X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor, col_dict = load_processed_data()
        train_network(net, X_train_tensor, Y_train_tensor, n_epochs=20, learning_rate=0.01)
        Q_io = calculate_importance(net)
        average_importance = torch.mean(Q_io, dim=1)
        sorted_indices = torch.argsort(average_importance, descending=True)
        top_12_indices = sorted_indices[:12].tolist()

        # Update the feature counter
        feature_counter.update(top_12_indices)

        net_weight_matrix = retrain_with_top_features(top_12_indices, X_train_tensor, Y_train_tensor)
        X_test_matrix_tensor = X_test_tensor[:, top_12_indices]
        weight_matrix_metrics = evaluate_model(net_weight_matrix, X_test_matrix_tensor, Y_test_tensor)

        for key, value in weight_matrix_metrics.items():
            metrics_accumulator[key] += value

    # Calculate average metrics
    for key in metrics_accumulator:
        metrics_accumulator[key] /= N_TRIALS

    # Get the most common 12 features over the iterations
    most_common_indices = [item[0] for item in feature_counter.most_common(12)]
    most_common_columns = [col_dict[i] for i in most_common_indices]

    print("Most common top 12 columns over 10 trials:", most_common_columns)
    print("use weight matrix method: the average metric result over 10 trials:")
    for key, value in metrics_accumulator.items():
        print(f"{key}: {value:.2f}")


