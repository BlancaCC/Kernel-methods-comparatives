###################################################################
# Kernel svm regression 
# Nested cross validation based on template_n_component
# Date end of September 2023 
###################################################################


from sklearn.compose import TransformedTargetRegressor
from utils.template_n_components import template_n_components

from sklearn.preprocessing import  StandardScaler
from sklearn.pipeline import Pipeline

from models_regression.params import function_param_grid_ksvm
from sklearn.svm import SVR

# utils 
import hyperparameters_config.name_of_pipeline as name_pipeline


def nested_kernel_SVR(X, y,
                            dataset_name:str, cv:int, n_jobs:int):
    model = 'Kernel SVR'

    dimension = X.shape[1]
    K = 5
    bias = -3
    base = 4

    param_grid = function_param_grid_ksvm(dimension, K, bias, base)
                    
    # Create the pipeline
    def get_inner_estimator(_):
        '''
        Function with the n_components as params that return the inner_estimator 
        created by a convenient pipeline and Transformation
        '''
        pipeline = Pipeline([
        (name_pipeline.scaler, StandardScaler()),
        (name_pipeline.kernel_svm, SVR(kernel='rbf') )
        ])   
        
        inner_estimator = TransformedTargetRegressor(regressor=pipeline,
                                                     transformer=StandardScaler())
        return inner_estimator
    
   
    results, cv_results = template_n_components(X, y,
                                dataset_name, cv, n_jobs, model,
                                param_grid, get_inner_estimator, 
                                True)
    return results, cv_results
   
     