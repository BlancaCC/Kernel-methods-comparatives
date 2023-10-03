###################################################################
# Nyström ridge regression 
# Nested cross validation based on template_n_component
# Date end of September 2023 
###################################################################

from sklearn.compose import TransformedTargetRegressor
from models_regression.params import function_param_grid_nystrom_ridge_regression
from utils.template_n_components import template_n_components
from sklearn.preprocessing import  StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge

# utils 
import hyperparameters_config.name_of_pipeline as name_pipeline


def nested_Nystrom_ridge_regression(X, y,
                            dataset_name:str, cv:int, n_jobs:int, X_test=False, y_test=False):
    model = 'Nystrom and ridge regression'

    dimension = X.shape[1]
    K = 5
    bias = -3
    base = 4

    param_grid = function_param_grid_nystrom_ridge_regression(dimension, K, bias, base)
                    
    # Create the pipeline
    def get_inner_estimator(n_components):
        '''
        Function with the n_components as params that return the inner_estimator 
        created by a convenient pipeline and Transformation
        '''
        # Create the Nystroem approximation
        nystrom = Nystroem(kernel='rbf', n_components= n_components)

        pipeline = Pipeline([
        (name_pipeline.scaler, StandardScaler()),
        (name_pipeline.nystrom, nystrom),
        (name_pipeline.ridge_regression, Ridge() )
        ])   
        
        inner_estimator = TransformedTargetRegressor(regressor=pipeline,
                                                     transformer=StandardScaler())
        return inner_estimator
    
   
    results, cv_results = template_n_components(X, y,
                                dataset_name, cv, n_jobs, model,
                                param_grid, get_inner_estimator, 
                                without_features=False, X_test=X_test, y_test=y_test)
    return results, cv_results
   
     