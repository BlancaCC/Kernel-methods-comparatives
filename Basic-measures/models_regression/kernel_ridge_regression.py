###################################################################
# Kernel ridge regression
# Nested cross validation based on template_n_component
# name of the main function: `nested_kernel_ridge_regression`
# Date end of August 2023 
###################################################################

from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from models_regression.params import function_param_grid_kernel_ridge_regression
from utils.template_n_components import template_n_components
import numpy as np
from sklearn.preprocessing import  StandardScaler
from sklearn.pipeline import Pipeline

# utils 
import hyperparameters_config.name_of_pipeline as name_pipeline


def nested_kernel_ridge_regression(X, y,
                            dataset_name:str, cv:int, n_jobs:int):
    model = 'Kernel ridge regression'

    dimension = X.shape[1]
    K = 5
    bias = -3
    base = 4

    param_grid = function_param_grid_kernel_ridge_regression(dimension, K, bias, base)
                    
    # Create the pipeline
    def get_inner_estimator(_):
        '''
        Function with the n_components as params that return the inner_estimator 
        created by a convenient pipeline and Transformation
        '''
        pipeline = Pipeline([
        (name_pipeline.scaler, StandardScaler()),
        (name_pipeline.kernel_ridge_regression, KernelRidge(kernel='rbf'))
        ])
        inner_estimator = TransformedTargetRegressor(regressor=pipeline,
                                                     transformer=StandardScaler())
      
        return inner_estimator
    
   
    results, cv_results = template_n_components(X, y,
                                dataset_name, cv, n_jobs, model,
                                param_grid, get_inner_estimator, 
                                True)
    return results, cv_results
   
     