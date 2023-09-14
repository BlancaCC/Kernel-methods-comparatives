###################################################################
# Fourier Random Features  SVR KF (k fold)
###################################################################
from utils.template_n_components import template_n_components
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import  StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import SVR

from sklearn.pipeline import Pipeline


# utils 
import hyperparameters_config.name_of_pipeline as name_pipeline


def Fourier_random_features_svr_KF(X, y,
                            dataset_name:str, cv:int, n_jobs:int):
    model = 'Fourier Random Features SVR'

    n_rows,dim = X.shape
    l_gamma_frf = np.logspace(-8, 0,num=4, base=2)/ dim
    l_C = np.logspace(0, 5,num=5, base=10)
    l_epsilon = np.logspace(-6, 2,num=4, base=2)

    reg = lambda name, hyperparameter : f'regressor__{name}__{hyperparameter}'

    param_grid ={ 
        reg(name_pipeline.fourier_random_features, 'gamma') : l_gamma_frf,
        reg(name_pipeline.svr, 'C') : l_C, 
        reg(name_pipeline.svr, 'epsilon') : l_epsilon  
    }
                
    # Create the pipeline
    def get_inner_estimator(n_components):
        '''
        Function with the n_components as params that return the inner_estimator 
        created by a convenient pipeline and Transformation
        '''
        pipeline = Pipeline([
                (name_pipeline.scaler, StandardScaler()),
                (name_pipeline.fourier_random_features, RBFSampler(n_components= n_components)),
                (name_pipeline.svr, SVR(kernel = 'linear'))
            ])
            
        inner_estimator = TransformedTargetRegressor(regressor=pipeline,
                                                transformer=StandardScaler())
        return inner_estimator
    
   
    
    results, cv_results = template_n_components(X, y,
                                dataset_name, cv, n_jobs, model,
                                param_grid, get_inner_estimator)
    return results, cv_results
   
     