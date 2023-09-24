###  The 3 feature selection methods mentioned in dataset paper
from NN10b.data_preprocessing import load_processed_data
from NN10b.feature_select_method import mrmr_method, compute_correlations, genetic_algorithm_method
from NN10b.metric import  train_evaluate_average

'''Rahman, J. S., Gedeon, T., Caldwell, S., Jones, R., & Jin, Z. (2021). Towards effective music therapy for mental health care using machine learning tools: human affective reasoning and music genres. Journal of Artificial Intelligence and Soft Computing Research, 11(1), 5-20.'''

## load data
X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor, col_dict = load_processed_data()
##
# Main execution
if __name__ == "__main__":
    X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor, col_dict = load_processed_data()

    # SD method
    train_evaluate_average(compute_correlations, "SD", X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, col_dict)

    # MRMR method
    train_evaluate_average(mrmr_method, "MRMR", X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, col_dict)

    # GA method
    train_evaluate_average(genetic_algorithm_method, "GA", X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, col_dict)
