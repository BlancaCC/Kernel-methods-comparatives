###################################################################
# Random Fourier Features and Ridge Regression 
# Nested cross validation based on template_n_component
# name of the main function: `nested_kernel_ridge_classification`
# Date end of September 2023 
###################################################################
from models_classification.params import function_param_grid_Fourier_random_features_ridge_classification
from utils.template_n_components import template_n_components

from sklearn.preprocessing import  StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import RidgeClassifier

# utils 
import hyperparameters_config.name_of_pipeline as name_pipeline


def nested_random_Fourier_features_ridge_classification(X, y,
                            dataset_name:str, cv:int, n_jobs:int, X_test=False, y_test=False):
    model = 'Random Fourier features and ridge classification'

    dimension = X.shape[1]
    K = 5
    bias = -3
    base = 4

    param_grid = function_param_grid_Fourier_random_features_ridge_classification(dimension, K, bias, base)
                    
    # Create the pipeline
    def get_inner_estimator(n_components):
        '''
        Function with the n_components as params that return the inner_estimator 
        created by a convenient pipeline and Transformation
        '''
        random_fourier_features = RBFSampler(n_components= n_components)

        pipeline = Pipeline([
        (name_pipeline.scaler, StandardScaler()),
        (name_pipeline.fourier_random_features, random_fourier_features),
        (name_pipeline.ridge_classification, RidgeClassifier(tol = 0.001) )
        ])   
        
        inner_estimator = pipeline # In classification no output transform is needed
        return inner_estimator
    
   
    results, cv_results = template_n_components(X, y,
                                dataset_name, cv, n_jobs, model,
                                param_grid, get_inner_estimator, 
                                without_features=False, X_test=X_test, y_test=y_test)
    return results, cv_results
   
     