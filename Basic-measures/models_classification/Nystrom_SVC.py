###################################################################
# Nyström and SVC
# Nested cross validation based on template_n_component
# Date end of September 2023 
###################################################################

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVC

from models_classification.params import function_param_grid_nystrom_SVC
from utils.template_n_components import template_n_components


# utils 
import hyperparameters_config.name_of_pipeline as name_pipeline

def nested_Nystrom_SVC(X, y,
                            dataset_name:str, cv:int, n_jobs:int):
    model = 'Nystrom and SVC'

    dimension = X.shape[1]
    K = 5
    bias = -3
    base = 4

    param_grid = function_param_grid_nystrom_SVC(dimension, K, bias, base)
                    
    # Create the pipeline
    def get_inner_estimator(n_components):
        '''
        Function with the n_components as params that return the inner_estimator 
        created by a convenient pipeline and Transformation
        '''

        pipeline = Pipeline([
        (name_pipeline.scaler, StandardScaler()),
        (name_pipeline.nystrom, Nystroem(kernel='rbf', n_components= n_components)),
        (name_pipeline.svm, SVC(kernel='linear'))
        ])   
        
        inner_estimator = pipeline # In classification no output transform is needed
        return inner_estimator
    
   
    results, cv_results = template_n_components(X, y,
                                dataset_name, cv, n_jobs, model,
                                param_grid, get_inner_estimator, 
                                without_features=False)
    return results, cv_results
   
     