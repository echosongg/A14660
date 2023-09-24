# A1 comp4660
### stdid: u7439250
### name: Xinni Song
Music EEG Feature Selection and Neural Network Analysis
This repository contains the implementation of various feature selection methods on EEG datasets related to music. The main objective is to identify the most relevant features from EEG data, and subsequently, use them to train a neural network model for classification tasks.

## Dataset
The dataset music-eeg-features.xlsx contains EEG features extracted during music sessions. The data has been preprocessed and saved in processed_music_data.xlsx.

## Structure
The codebase is organized into various modules, each serving a specific purpose:

data_preprocessing.py: Contains methods for loading and preprocessing the data.
feature_select_method.py: Implements various feature selection methods such as Pearson correlations (compute_correlations), MRMR method (mrmr_method), and a method based on genetic algorithms (genetic_algorithm_method).
Used to compare with weight matrix analysis.
feature_selection_in_dataset_paper.py: Main script that loads the data, selects features, and evaluates the performance of the neural network on the selected features.
feature_selection_matrix.py: Provides matrix-based feature selection methods.
metric.py: Contains methods for evaluating the performance of the neural network using various metrics.
netrual_network_structure.py: Defines the structure of the neural network and provides training routines.
visualize.py: Offers visualization methods for feature importance.
Feature Selection Methods
We implemented three feature selection methods:

SD (Standard Deviation) Method: Based on Pearson correlation.
MRMR (Minimum Redundancy Maximum Relevance) Method: Uses mutual information for feature selection.
Weight Matrix Analysis Method: Utilizes the importance of weights in a neural network to select features.
## Evaluation Metrics
The performance of the neural network is evaluated using the following metrics:

Accuracy
Precision
Recall
F1 Score
Specificity
Geometric Mean
## Visualization
We provide visualizations to help interpret the results. These visualizations include:

Bar charts representing the most frequently selected features across multiple runs.
Tables summarizing the performance metrics of the neural network.

## How to Run
Ensure the necessary libraries installed.
Load the dataset using the load_processed_data function from data_preprocessing.py.
Run the feature_selection_in_dataset_paper.py and feature_selection_matrix.py to perform feature selection and evaluate the neural network's performance on the selected features.
Use the visualize.py script for visualizations.