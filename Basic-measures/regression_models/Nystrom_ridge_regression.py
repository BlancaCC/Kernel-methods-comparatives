###################################################################
# Nyström Ridge Regression 
###################################################################
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import  StandardScaler
from sklearn.kernel_approximation import Nystroem

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib

from sklearn.metrics import r2_score


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
    n_rows, dim = X.shape
    l_alpha = np.array([10.**k for k in range(-3, 5, 1)])
    l_gamma = np.logspace(-4, 10,6, base=2)/ dim
    param_grid ={
                'regressor__nystrom__gamma': l_gamma,
                'regressor__ridge__alpha': l_alpha}
    hyperparameters_to_test = len(l_alpha) * len(l_gamma) 

    # Number from random features
    percent = [0.5, 1, 2,4,6,8,10,14,16,18,20,22,24,26,28,30]
    n_components_list = list(map(lambda x: int(x* n_rows / 100), percent ))

    # print information
    head_title = f'''
    {'-'*20}
    'Model: {model} 
    \tDataset: {dataset_name}  of shape {X.shape} \tCV: {cv} \tn_jobs: {n_jobs}
    \t params: { param_grid} 
    \t number of hyperparameter {hyperparameters_to_test}
    \t n components {n_components_list}
    \t which correspond with the {percent} % of the data
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

        # Create the Nystroem approximation
        nystrom = Nystroem(kernel='rbf', n_components= n_components)

        # Create the pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('nystrom', nystrom),
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