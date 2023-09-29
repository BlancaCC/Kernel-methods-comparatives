###################################################################
# Random Fourier Features and SVR
# Nested cross validation based on template_n_component
# Date end of September 2023 
###################################################################


from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import  SVR

from models_regression.params import function_param_grid_random_Fourier_features_svr
from utils.template_n_components import template_n_components
from sklearn.preprocessing import  StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler


# utils 
import hyperparameters_config.name_of_pipeline as name_pipeline


def nested_random_Fourier_features_SVR(X, y,
                            dataset_name:str, cv:int, n_jobs:int):
    model = 'Random Fourier features and SVR'

    dimension = X.shape[1]
    K = 5
    bias = -3
    base = 4

    param_grid = function_param_grid_random_Fourier_features_svr(dimension, K, bias, base)
                    
    # Create the pipeline
    def get_inner_estimator(n_components):
        '''
        Function with the n_components as params that return the inner_estimator 
        created by a convenient pipeline and Transformation
        '''

        pipeline = Pipeline([
        (name_pipeline.scaler, StandardScaler()),
        (name_pipeline.fourier_random_features, RBFSampler(n_components= n_components)),
        (name_pipeline.svm, SVR(kernel='linear') )
        ])   
        
        inner_estimator = TransformedTargetRegressor(regressor=pipeline,
                                                     transformer=StandardScaler())
        return inner_estimator
    
   
    results, cv_results = template_n_components(X, y,
                                dataset_name, cv, n_jobs, model,
                                param_grid, get_inner_estimator, 
                                without_features=False)
    return results, cv_results
   
     