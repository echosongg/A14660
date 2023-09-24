## caculate input importance for output， by technology paper
import torch
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import numpy as np
from scipy.stats import pearsonr
from deap import base, creator, tools, algorithms
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

## the method I want to vertify
def calculate_importance(net):
    # Calculate P for hidden-output connection
    w_ho = net.output_layer.weight.data
    P_ho = torch.abs(w_ho) / torch.sum(torch.abs(w_ho), dim=1, keepdim=True)

    # Calculate P for input-hidden connection
    w_ih = net.input_layer.weight.data
    P_ih = torch.abs(w_ih) / torch.sum(torch.abs(w_ih), dim=1, keepdim=True)

    # Calculate Q for each input-output pair
    Q_io = torch.zeros(P_ih.shape[1], P_ho.shape[0])
    for k in range(P_ho.shape[0]):
        for i in range(P_ih.shape[1]):
            Q_io[i, k] = torch.sum(P_ih[:, i] * P_ho[k, :])

    return Q_io

## the method mentioned in dataset

def compute_correlations(X, y, n_features=12):
    """
    Compute Pearson correlations between each column of X and y.

    Args:
    - X (array-like): A 2D array with shape (n_samples, n_features).
    - y (array-like): A 1D array with shape (n_samples,).

    Returns:
    - List of indices of the top n_features columns in X with highest absolute correlation with y.
    """
    correlations = [pearsonr(X[:, i], y)[0] for i in range(X.shape[1])]

    # Get indices of top n_features correlations (by absolute value)
    top_indices = sorted(range(len(correlations)), key=lambda i: abs(correlations[i]), reverse=True)[:n_features]

    return top_indices

def mrmr_method(X, y, n_features=12):
    n_samples, n_features_all = X.shape
    feature_indices = list(range(n_features_all))

    # Compute mutual information between each feature and the target
    relevance = mutual_info_classif(X, y)

    # Start with the feature that has the highest mutual information with the target
    selected_features = [np.argmax(relevance)]
    feature_indices.remove(selected_features[0])

    while len(selected_features) < n_features:
        # Compute average redundancy between selected features and remaining features
        redundancy_matrix = np.zeros((len(selected_features), len(feature_indices)))
        for i, selected in enumerate(selected_features):
            for j, remaining in enumerate(feature_indices):
                # Use mutual_info_regression for feature-feature interaction
                redundancy_matrix[i, j] = mutual_info_regression(X[:, [selected]], X[:, remaining])[0]
        avg_redundancy = np.mean(redundancy_matrix, axis=0)

        # Compute mRMR scores
        mrmr_scores = relevance[feature_indices] - avg_redundancy

        # Select the feature with the highest mRMR score
        next_selected = feature_indices[np.argmax(mrmr_scores)]
        selected_features.append(next_selected)
        feature_indices.remove(next_selected)

    return selected_features

def evaluate(individual, X, y):
    """Evaluate the accuracy of a classifier based on the selected features."""
    selected_features = [i for i, selected in enumerate(individual) if selected]
    if len(selected_features) == 0:  # Avoid empty selection
        return (0,)
    clf = MLPClassifier(max_iter=1000)  # Simple neural network classifier
    clf.fit(X[:, selected_features], y)
    predictions = clf.predict(X[:, selected_features])
    return (accuracy_score(y, predictions),)

def genetic_algorithm_method(X, y, n_features=12):
    # Define types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.choice, 2, p=[0.2, 0.8])  # 80% probability to start with a feature
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, X=X, y=y)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=50)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)

    # Get the best individual from the final population
    best_ind = tools.selBest(population, k=1)[0]
    selected_features = [i for i, selected in enumerate(best_ind) if selected]
    return selected_features[:n_features]