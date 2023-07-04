import time
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from get_data import get_data
from params import param_grid_ksvm
from config import path

# Parse command-line arguments

parser = argparse.ArgumentParser(description='Model training and evaluation')
parser.add_argument('dataset', type=str, help='Name of the dataset')
parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs (default: 1)')
parser.add_argument('--cv', type=int, default=5, help='Number of CV splits (default: 5)')
args = parser.parse_args()

model = 'KSVM'
output_file = path + f'{model}_{args.dataset}_cv_{args.cv}.txt'

head_title = f'''
{'-'*20}
'Model: {model} \tDataset: {args.dataset} \tCV: {args.cv} \tn_jobs: {args.n_jobs}'
{'-'*20}
'''
print(head_title)


# Get Data
X_train, y_train, X_test, y_test = get_data(args.dataset)
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
param_grid = param_grid_ksvm

# Create the GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=args.cv, n_jobs=args.n_jobs, verbose=3)
# Fit the grid search on the data
grid_search.fit(X_train, y_train)

# Retrain with best parameters
svm = SVC(kernel='rbf', C=grid_search.best_params_['svm__C'], gamma=grid_search.best_params_['svm__gamma'])
start_time = time.time()
svm.fit(X_train, y_train)
training_time = time.time() - start_time

# Predict with X_test
y_pred = svm.predict(X_test)
# Calculate the accuracy score on the test data
accuracy = accuracy_score(y_test, y_pred)

# Print the best parameters and accuracy score
print('Final scores')
print("Best Parameters: ", grid_search.best_params_)
print("Best Accuracy in CV: ", grid_search.best_score_)
print("Training Time: ", training_time)
print("Accuracy in test: ", accuracy)

# Write the results to a file

with open(output_file, 'w') as f:
    f.write(head_title)
    f.write(f"Best Parameters: {grid_search.best_params_}\n")
    f.write(f"Best Accuracy in CV: {grid_search.best_score_}\n")
    f.write(f"Training Time: {training_time}\n")
    f.write("Accuracy in test: {accuracy}\n")

print(f"Results written to {output_file}")







