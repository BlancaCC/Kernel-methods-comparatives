
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

from sklearn.svm import SVC
from get_data import get_data
from params import function_param_grid_ksvm
from config import path, path_for_joblib

# Parse command-line arguments

parser = argparse.ArgumentParser(description='Model training and evaluation')
parser.add_argument('dataset', type=str, help='Name of the dataset')
parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs (default: 1)')
parser.add_argument('--cv', type=int, default=5, help='Number of CV splits (default: 5)')
args = parser.parse_args()

model = 'KSVM classification'
file_model_name = 'KSVM_classification'
file_model_name_arg = f'{file_model_name}_{args.dataset}_cv_{args.cv}'
output_file = path + file_model_name_arg + '.txt'


# Get Data
X_train, y_train, X_test, y_test = get_data(args.dataset)

dimension = X_train.shape[1]
K = 5
bias = -3
base = 4

param_grid = function_param_grid_ksvm(dimension, K, bias, base)

# print information
head_title = f'''
{'-'*20}
'Model: {model} 
\tDataset: {args.dataset} \tCV: {args.cv} \tn_jobs: {args.n_jobs}
\t params: { param_grid}
{'-'*20}
'''
print(head_title)

## Data preprocessing

# Create the scaler
scaler = StandardScaler()

# Create the SVM classifier
svm = SVC(kernel='rbf')

# Create the pipeline
pipeline = Pipeline([
    ('scaler', scaler),
    ('svm', svm)
])

# Define the parameter grid
# Create the GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=args.cv, n_jobs=args.n_jobs)
# Fit the grid search on the data
grid_search.fit(X_train, y_train)

# Get the best pipeline from the grid search
best_pipeline = grid_search.best_estimator_

# Predict with X_test
y_pred = best_pipeline.predict(X_test)
# Calculate the accuracy score on the test data
accuracy = accuracy_score(y_test, y_pred)

# Print the best parameters and accuracy score
print('Final scores')
print("Best Parameters: ", grid_search.best_params_)
print("Best Accuracy in CV: ", grid_search.best_score_)
print("Training Time: ", grid_search.refit_time_)
print("Accuracy in test: ", accuracy)

# Write the results to a file


with open(output_file, 'w') as f:
    f.write(head_title)
    f.write(f"Best Parameters: {grid_search.best_params_}\n")
    f.write(f"Best Accuracy in CV: {grid_search.best_score_}\n")
    f.write(f"Training Time: {grid_search.refit_time_}\n")
    f.write(f"Accuracy in test: {accuracy}\n")
    f.write(str(grid_search.cv_results_))

print(f"Results written to {output_file}")

# Save the grid search object to a file
joblib.dump(grid_search, path_for_joblib+f'grid_search_{file_model_name_arg}.pkl')







