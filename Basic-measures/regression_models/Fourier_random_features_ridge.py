###################################################################
# Fourier Random Features + Ridge Regression 
###################################################################
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import  StandardScaler
from sklearn.kernel_approximation import RBFSampler

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib

from sklearn.metrics import r2_score


from config import path_for_dataset
from utils.nested_cross_validation import nested_cross_validation

def Fourier_random_features_ridge_regression_KF(X, y,
                            dataset_name:str, cv:int, n_jobs:int,
                            verbose=True, save_mode=True):
    model = 'Fourier Random Features ridge regression'
    file_model_name = 'Fourier_random_features_ridge_regression'
    file_model_name_arg = f'{file_model_name}_{dataset_name}_cv_{cv}'
    output_file_accuracy, output_file_verbose = path_for_dataset(dataset_name) 
    output_file_verbose += file_model_name_arg 
    output_file_accuracy += file_model_name_arg + '.csv'

    # Hyperparameters for CV
    dim = X.shape[1]
    l_alpha     = [10.**k for k in range(-7, 2)] 
    l_gamma = list( np.array([2.**k for k in range(-17, 2)]) / dim)
    param_grid ={
                'regressor__fourier_random_features__gamma': l_gamma,
                'regressor__ridge__alpha': l_alpha}
    hyperparameters_to_test = len(l_alpha) * len(l_gamma) 

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
    cv_results_total = { 
        "n_components":[],
        "Score in test":[],
        "Training Time":[],
        "Best Parameters": [],
        "Best Score in CV": [], 
    }
    results_total = {
        'n_components' :[],
        "Mean Score in test" :[],
        "Std Score in test" :[],
        "Mean Training Time": [],
        "Std Training Time": [],
        "Mean Best Score in CV": [],
        "Std Best Score in CV": [] 
    }

    for n_components in n_components_list:
        print('-'*20+f'\nNumber of components for this iteration: {n_components}')
 
        results_total['n_components'].append(n_components)
        cv_results_total['n_components'] += [n_components for _ in range(4)] # una por cada nested cross validation

        # Create the pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('fourier_random_features', RBFSampler(n_components= n_components)),
            ('ridge', Ridge())
        ])
        
        inner_estimator = TransformedTargetRegressor(regressor=pipeline,
                                                transformer=StandardScaler())
        
        # Define the parameter grid
        # Create the GridSearchCV object
        grid_search = GridSearchCV(inner_estimator, param_grid, cv=cv, n_jobs=n_jobs, refit=True)
        # Nested cross validation
        results, cv_results = nested_cross_validation(X,y, grid_search, 4, r2_score)

        for key in results.keys():
            results_total[key] += results[key]
        
        for key in cv_results.keys():
            cv_results_total[key] += cv_results[key]
        

        print(results)
        print(cv_results)

    if save_mode:
        print('Resultados finales')
        print(results_total)
        print(cv_results_total)

        results_df = pd.DataFrame(results_total)
        results_df.to_csv(output_file_accuracy, index=False)
        results_cv_df = pd.DataFrame(cv_results_total)
        results_cv_df.to_csv(output_file_verbose+ '.csv', index=True)

        file = open(output_file_verbose+ '.txt', "w")
        file.write(str(head_title) + "\n")
        file.write(str(results_total) + "\n")
        file.write(str(cv_results_total) + "\n")
        file.close()

    return results, cv_results