################################################################
#               Neural Networks
################################################################

import argparse
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

from get_data import get_data
from params import n_components_list
from config import path, path_for_joblib


from params import get_neural_networks_sizes, function_param_grid_neural_classification

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Model training and evaluation')
parser.add_argument('dataset', type=str, help='Name of the dataset')
parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs (default: 1)')
parser.add_argument('--cv', type=int, default=5, help='Number of CV splits (default: 5)')
args = parser.parse_args()

model = 'Neural Networks'
file_model_name = 'neural_network_classification'
file_model_name_arg = f'{file_model_name}_{args.dataset}_cv_{args.cv}'
output_file = path + file_model_name_arg + '.csv'

# Get Data
X_train, y_train, X_test, y_test = get_data(args.dataset)

size,dimension = X_train.shape
K = 6
bias = 1
base = 10
num = 14
param_grid = function_param_grid_neural_classification(K, bias,num, base)

percent = 1.2
maximum_layers=3
maximum_units = 20000
neural_networks_sizes = get_neural_networks_sizes(size, percent, maximum_units, maximum_layers)

# print information 
head_title = f'''
{'-'*20}
Model: {model} 

\tDataset: {args.dataset} \tCV: {args.cv} \tn_jobs: {args.n_jobs}
\tnumber of components to test: {n_components_list}
\tparam_grid (k= {K}, bias = {5}, dimension = {dimension} base = {base}) = \n{param_grid}
\t neural networks sizes: {neural_networks_sizes} 
\t size {size} percent {percent} maximum layers {3}
{'-'*20}
'''
print(head_title)

## Data preprocessing
results = []
for hidden_layer_sizes in neural_networks_sizes:
    # Create Neural Network models 
    mlp = MLPClassifier(hidden_layer_sizes, 
                                max_iter=10000)

    # Create the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp_classification', mlp)
    ])

    # Create the GridSearchCV object
    grid_search = GridSearchCV(pipeline, param_grid, cv=args.cv, n_jobs=args.n_jobs, refit=True)
    # Fit the grid search on the data
    grid_search.fit(X_train, y_train)

    # Get the best pipeline from the grid search
    best_pipeline = grid_search.best_estimator_

    # Predict with X_test
    y_pred = best_pipeline.predict(X_test)

    # Calculate the accuracy score on the test data
    accuracy = accuracy_score(y_test, y_pred)

    # Print the best parameters and accuracy score
    print(f'Final scores hidden layer sizes: {hidden_layer_sizes}')
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Accuracy in CV: ", grid_search.best_score_)
    print("Training Time: ", grid_search.refit_time_)
    print("Accuracy in test: ", accuracy)

    # Store the results in a dictionary
    result = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'best_params': grid_search.best_params_,
        'best_accuracy_cv': grid_search.best_score_,
        'training_time': grid_search.refit_time_,
        'accuracy_test': accuracy, 
        **grid_search.cv_results_ 
    }

    # Append the result to the results list
    results.append(result)

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)

# Write the results to a file
results_df.to_csv(output_file, index=False)

## save the model 

# Save the grid search object to a file
joblib.dump(grid_search, path_for_joblib+f'grid_search_{file_model_name_arg}.pkl')

print(f"Results written to {output_file}")
