###################################################################
# Nyström Ridge Regression 
###################################################################
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib

from sklearn.metrics import r2_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

from config import path_for_dataset
from utils.nested_cross_validation import nested_cross_validation

def Nystrom_ridge_regression_KF(X, y,
                            dataset_name:str, cv:int, n_jobs:int,
                            verbose=True, save_mode=True):
    model = 'Nysytom ridge regression'
    file_model_name = 'nystrom_ridge_regression'
    file_model_name_arg = f'{file_model_name}_{dataset_name}_cv_{cv}'
    output_file_accuracy, output_file_verbose = path_for_dataset(dataset_name) 
    output_file_verbose += file_model_name_arg 
    output_file_accuracy += file_model_name_arg + '.csv'

    # Hyperparameters for CV
    dim = X.shape[1]
    l_C     = [10.**k for k in range(-3, 4)] 
    l_gamma = list( np.array([2.**k for k in range(-2, 7)]) / dim)
    l_epsilon = [2.**k for k in range(-6, 0)]

    param_grid ={'regressor__svr__C': l_C,
                'regressor__svr__gamma': l_gamma,
                'regressor__svr__epsilon': l_epsilon}
    hyperparameters_to_test = len(l_C) * len(l_gamma) * l_epsilon
    # Number from random features
    n_components_list = [10, 20, 50, 100, 200, 500, 1000]

    # print information
    head_title = f'''
    {'-'*20}
    'Model: {model} 
    \tDataset: {dataset_name}  of shape {X.shape} \tCV: {cv} \tn_jobs: {n_jobs}
    \t params: { param_grid} 
    \t number of hyperparameter {hyperparameters_to_test}
    \t n components {n_components_list}
    {'-'*20}
    '''
    print(head_title)
    # Create the pipeline
    svr = SVR(kernel='rbf', 
          shrinking=False, 
          tol=1.e-3)
    pipeline = Pipeline(steps=[('minmax_sc', MinMaxScaler()),
                       ('svr', svr)])
    inner_estimator = TransformedTargetRegressor(regressor=pipeline,
                                             transformer=StandardScaler())

    
    # Define the parameter grid
    # Create the GridSearchCV object
    grid_search = GridSearchCV(inner_estimator, param_grid, cv=cv, n_jobs=n_jobs, refit=True)
    # Nested cross validation
    results, cv_results = nested_cross_validation(X,y, grid_search, 4, r2_score)
    
    print(results)
    print(cv_results)

    if save_mode:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file_accuracy, index=False)
        results_cv_df = pd.DataFrame(cv_results)
        results_cv_df.to_csv(output_file_verbose+ '.csv', index=True)

        file = open(output_file_verbose+ '.txt', "w")
        file.write(str(head_title) + "\n")
        file.write(str(results) + "\n")
        file.write(str(cv_results) + "\n")
        file.close()

    return results, cv_results