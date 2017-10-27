import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

dummy_variables = []  # List to store dummy variables and interactions involving two dummy variables


def get_predicted_value(x_row, weights):  # Function to return the predicted Y value
    predict_value = 0
    for i in range(len(weights)):
        predict_value += x_row[i] * weights[i]
    return predict_value


def get_error(x_row, y_actual, weights):  # Function to return the error - difference in actual and predicted values
    return y_actual - get_predicted_value(x_row, weights)


# Function to compute the optimal weight vector
def get_optimal_weight_vector(data_x, data_y, learning_rate_lambda):
    lambda_i = learning_rate_lambda * np.identity(data_x.shape[1])
    x_t_x = np.matmul(data_x.T, data_x)
    x_t_y = np.matmul(data_x.T, data_y)
    return np.matmul(np.linalg.inv(lambda_i + x_t_x), x_t_y)


# Function to compute the overall squared error of the model
def get_overall_squared_error(data_x, data_y, weights):
    total_error = 0
    for i in range(len(data_x)):
        total_error += (get_error(data_x[i], data_y[i], weights) ** 2)
    return total_error


# DUMMY VARIABLES FOR CLASSIFICATION ATTRIBUTES
def add_dummy_variables(train_data):
    train_data = train_data.copy()
    classification_features = list(train_data.dtypes[train_data.dtypes == 'object'].index)
    for feature in classification_features:
        feature_values = train_data[feature].unique()
        for elem in feature_values:
            col_name = str(feature + '_' + str(elem))
            train_data[col_name] = pd.Series(train_data[feature] == elem, dtype=int, index=train_data.index)
            dummy_variables.append(col_name)
        del train_data[feature]
    return train_data


# NORMALIZE DATA
def normalize_data(data):
    data = data.copy()
    numeric_features = list(data.dtypes[data.dtypes != 'object'].index)
    for feature in numeric_features:
        if feature in dummy_variables:
            continue
        mean = data[feature].mean()
        std = data[feature].std()
        if std == 0:
            del data[feature]
        else:
            data[feature] = data[feature] - mean
            data[feature] = data[feature] / std
    return data


# ADD QUADRATIC INTERACTION TERMS
def add_interaction_terms(data):
    length = data.shape[1]
    for column_i in range(length):
        for column_j in range(column_i + 1, length):
            col_name = str(data.columns[column_i]) + '_' + str(data.columns[column_j])
            data[col_name] = data.iloc[:, column_i] * data.iloc[:, column_j]
            if column_i in dummy_variables and column_j in dummy_variables:
                dummy_variables.append(col_name)
    return data


# DATA PRE-PROCESSING - SPLITTING TRAIN AND TEST
def pre_processing(data):
    # Remove unnecessary columns
    data.drop(['id', 'PlayerName', 'Country', 'GP_greater_than_0', 'sum_7yr_TOI', 'Overall'], axis=1, inplace=True)

    train_data_x = data[data['DraftYear'].isin([2004, 2005, 2006])]
    test_data_x = data[data['DraftYear'] == 2007]

    del train_data_x['DraftYear']
    del test_data_x['DraftYear']

    # Take Y vectors
    train_y = train_data_x['sum_7yr_GP']
    test_y = test_data_x['sum_7yr_GP']
    del train_data_x['sum_7yr_GP']
    del test_data_x['sum_7yr_GP']

    train_data_x = add_dummy_variables(train_data_x)
    train_data_x = add_interaction_terms(train_data_x)
    train_data_x = normalize_data(train_data_x)
    train_data_x['Constant'] = pd.Series([1 for _ in range(len(train_data_x))], dtype=int, index=train_data_x.index)

    test_data_x = add_dummy_variables(test_data_x)
    test_data_x = add_interaction_terms(test_data_x)
    test_data_x = normalize_data(test_data_x)
    test_data_x['Constant'] = pd.Series([1 for _ in range(len(test_data_x))], dtype=int, index=test_data_x.index)

    return train_data_x, train_y, test_data_x, test_y


# Function to drop if column's std is 0
def drop_columns_std_zero(data):
    for column in data.columns:
        if np.std(data[column]) == 0:
            del data[column]
    return data


# Function to create K folds in the cross validation - in random fashion
def create_k_folds(data_x, data_y, folds=10):
    each_fold_size = int(len(data_x) / folds)
    x_copy = list(data_x)
    y_copy = list(data_y)
    splits_x = []
    splits_y = []
    for i in range(folds):
        fold_x = []
        fold_y = []
        for _ in range(each_fold_size):
            idx = random.randrange(len(x_copy))
            fold_x.append(x_copy.pop(idx))
            fold_y.append(y_copy.pop(idx))
        splits_x.append(fold_x)
        splits_y.append(fold_y)
    return splits_x, splits_y


# Function to perform k fold cross validation
def do_k_fold_cross_validation(data_x, data_y, lambda_learning_rate, folds=10):
    x_folds, y_folds = create_k_folds(data_x, data_y, folds)
    errors = []
    for i in range(folds):
        train_set_x = []
        train_set_y = []
        test_set_x = x_folds[i]
        test_set_y = y_folds[i]
        for j in range(folds):
            if i == j:
                continue
            else:
                for k in range(len(x_folds[j])):
                    train_set_x.append(x_folds[j][k])
                    train_set_y.append(y_folds[j][k])
        weights = get_optimal_weight_vector(np.array(train_set_x), np.array(train_set_y), lambda_learning_rate)
        errors.append(get_overall_squared_error(test_set_x, test_set_y, weights))
    return errors


# MAIN LINEAR REGRESSION
def do_grid_search_linear_regression(data, lambda_rates):
    train_data_x, train_data_y, test_data_x, test_data_y = pre_processing(data)

    # Convert Data frame into matrices for computation
    train_data_x_matrix = np.array(train_data_x)
    train_data_y_matrix = np.array(train_data_y)
    test_data_x_matrix = np.array(test_data_x)
    test_data_y_matrix = np.array(test_data_y)

    # EVALUATION
    errors_train = []
    errors_test = []
    min_error_train = float('inf')
    min_error_test = float('inf')
    min_lambda_train = lambda_rates[0]
    min_lambda_test = lambda_rates[0]
    for learning_rate in lambda_rates:
        error = np.mean(do_k_fold_cross_validation(train_data_x_matrix, train_data_y_matrix, learning_rate))
        errors_train.append(error)
        if error < min_error_train:
            min_error_train = error
            min_lambda_train = learning_rate
        weights_for_test_learnt = get_optimal_weight_vector(train_data_x_matrix, train_data_y_matrix, learning_rate)
        test_error = get_overall_squared_error(test_data_x_matrix, test_data_y_matrix, weights_for_test_learnt)
        errors_test.append(test_error)
        if test_error < min_error_test:
            min_lambda_test = learning_rate
            min_error_test = test_error

    print('Best Lambda value by 10 fold cross validation = ' + str(min_lambda_train))
    print('Error at best lambda during 10 fold cross validation = ' + str(min_error_train))
    print('Best Lambda for Test Set = ' + str(min_lambda_test))
    print('Error at best lambda for Test = ' + str(min_error_test))

    # PLOT CURVE
    plt.figure(1)
    plt.semilogx(lambdas[1:], errors_train[1:], color='red', label='Cross Validation')
    plt.semilogx(min_lambda_train, min_error_train, marker='o', color='r', label="Best Train Lambda")
    plt.semilogx(lambdas[1:], errors_test[1:], color='green', label='Test Set')
    plt.semilogx(min_lambda_test, min_error_test, marker='x', color='r', label="Best Test Lambda")
    plt.legend()
    plt.title('Lambda vs Error')
    plt.xlabel('Lambda')
    plt.ylabel('Squared Error')

    plt.show()


if __name__ == "__main__":
    random.seed(0)
    data_set = pd.read_csv('preprocessed_datasets.csv')  # Load data
    lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
    do_grid_search_linear_regression(data_set, lambdas)
