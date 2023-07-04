################################################################
# Nyström + Ridge classification
######################################################

import time
import argparse
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from get_data import get_data
from params import n_components_list
from config import path
from params import function_param_grid_nystrom_ridge_classification


# Parse command-line arguments

parser = argparse.ArgumentParser(description='Model training and evaluation')
parser.add_argument('dataset', type=str, help='Name of the dataset')
parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs (default: 1)')
parser.add_argument('--cv', type=int, default=5, help='Number of CV splits (default: 5)')
args = parser.parse_args()

model = 'Nyström + Ridge Classification'
output_file = path + f'{model}_{args.dataset}_cv_{args.cv}.txt'


# Get Data
X_train, y_train, X_test, y_test = get_data(args.dataset)

# hyperparameter
dimension = X_train.shape[1]
K = 5
bias = -3
base = 4

param_grid = function_param_grid_nystrom_ridge_classification(dimension, K, bias, base)

# print information 
head_title = f'''
{'-'*20}
Model: {model} 

\tDataset: {args.dataset} \tCV: {args.cv} \tn_jobs: {args.n_jobs}
\nnumber of components to test: {n_components_list}
\nparam_grid (k= {K}, bias = {5}, dimension = {dimension} base = {base}) = \n{param_grid}
{'-'*20}
'''
## Data preprocessing
results = []
for n_components in n_components_list:
    # Create the scaler
    scaler = StandardScaler()

    # Create the Nystroem approximation
    nystroem = Nystroem(kernel='rbf', n_components= n_components)

    # Create the Ridge classifier
    ridge = RidgeClassifier()

    # Create the pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('nystroem', nystroem),
        ('ridge_classification', ridge)
    ])

    # Create the GridSearchCV object
    grid_search = GridSearchCV(pipeline, param_grid, cv=args.cv, n_jobs=args.n_jobs, verbose=3)
    # Fit the grid search on the data
    grid_search.fit(X_train, y_train)

    # Retrain with best parameters
    # Create the Nystroem approximation
    nystroem = Nystroem(kernel='rbf', gamma=grid_search.best_params_['nystroem__gamma'],  n_components= n_components)

    # Create the Ridge classifier
    ridge = RidgeClassifier(alpha= grid_search.best_params_['ridge_classification__alpha'])

    # Create the pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('nyström', nystroem),
        ('ridge_regression', ridge)
    ])

    start_time = time.time()
    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Predict with X_test
    y_pred = pipeline.predict(X_test)

    # Calculate the accuracy score on the test data
    accuracy = accuracy_score(y_test, y_pred)

    # Print the best parameters and accuracy score
    print(f'Final scores for n componentes: {n_components}')
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Accuracy in CV: ", grid_search.best_score_)
    print("Training Time: ", training_time)
    print("Accuracy in test: ", accuracy)

    # Store the results in a dictionary
    result = {
        'n_components': n_components,
        'best_params': grid_search.best_params_,
        'best_accuracy_cv': grid_search.best_score_,
        'training_time': training_time,
        'accuracy_test': accuracy
    }

    # Append the result to the results list
    results.append(result)

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)

# Write the results to a file
results_df.to_csv(output_file, index=True)

print(f"Results written to {output_file}")


