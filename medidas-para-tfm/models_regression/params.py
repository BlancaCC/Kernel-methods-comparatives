######################################################################
# Criterium for KSVM
# Some criterium in orden to choose hyperparameters
# C and gamma KSVM (18 y 19 de las notas se discute cómo gestionar C y gamma en clasificación)
# 25 el épsilon en regresión
# https://github.com/joseDorronsoro/SVMs-for-Classification-and-Regression/blob/master/notes_svm_regression_regression_2021.pdf 

import numpy as np
import hyperparameters_config.name_of_pipeline as name_pipeline

######################################################################
#                        For  KSVM             
######################################################################
def function_param_grid_ksvm(dimension:int, K:int, bias: int, base: int):
    start = -K
    end = K + bias 
    num = 5
    gamma_space = np.logspace(start, end, num, base=base) / dimension

    return {
        f'regressor__{name_pipeline.kernel_svm}__C' : np.logspace(-2, 4, 7, base=10), # [0.1, 1, 10, 100, 1000]
        f'regressor__{name_pipeline.kernel_svm}__gamma': gamma_space,
        
    }


######################################################################
#                        For  Kernel ridge regression     
######################################################################

def function_param_grid_kernel_ridge_regression(dimension:int, K:int, bias: int, base: int, num = 6):
    start = -K
    end = K + bias 
    gamma_space = np.logspace(start, end, num, base=base) / dimension

    return {
        f'regressor__{name_pipeline.kernel_ridge_regression}__gamma' : gamma_space,
        f'regressor__{name_pipeline.kernel_ridge_regression}__alpha' : np.logspace(-4, 2, num, base=10)
      }

# Number from random features
n_components_list = [10, 20, 50, 100, 200, 500, 1000]


######################################################################
#                        For  Nyström + Ridge regression              
######################################################################

def function_param_grid_nystrom_ridge_regression(dimension:int, K:int, bias: int, base: int):
    start = -K
    end = K + bias 
    num = 5
    gamma_space = np.logspace(start, end, num, base=base) / dimension

    return {
        f'regressor__{name_pipeline.nystrom}__gamma' : gamma_space,
        f'regressor__{name_pipeline.ridge_regression }__alpha' : np.logspace(-4, 2, 5, base=10)
      }


######################################################################
#                        For  Nyström + SVM regression              
######################################################################

def function_param_grid_nystrom_SVR(dimension:int, K:int, bias: int, base: int,  num:int=5):
    start = -K
    end = K + bias 
    gamma_space = np.logspace(start, end, num, base=base) / dimension
    return {
        f'regressor__{name_pipeline.nystrom}__gamma' : gamma_space,
        f'regressor__{name_pipeline.svm}__C' : np.logspace(-4, 2, num, base=10)
    }



######################################################################
#                        For Fourier Random Features + Ridge regression              
######################################################################

def function_param_grid_Fourier_random_features_ridge_regression(dimension:int, K:int, bias: int, base: int, num = 5):
    start = -K
    end = K + bias 
    gamma_space = np.logspace(start, end, num, base=base) / dimension
    return {
        f'regressor__{name_pipeline.fourier_random_features}__gamma' : list(gamma_space),
        f'regressor__{name_pipeline.ridge_regression }__alpha' : np.logspace(-4, 2, num, base=10)
    }



######################################################################
#                For Random Fourier Features + SVM regression              
######################################################################

def function_param_grid_random_Fourier_features_svr(dimension:int, K:int, bias: int, base: int,  num:int=6):
    start = -K
    end = K + bias 
    gamma_space = np.logspace(start, end, num, base=base) / dimension
    return {
        f'regressor__{name_pipeline.fourier_random_features}__gamma' : list(gamma_space),
        f'regressor__{name_pipeline.svm}__C' : np.logspace(-4, 2, num, base=10)
    }


#############################################################################
#                   Neural Network regression 
#############################################################################
def get_neural_networks_sizes(size: int, percent:float, maximum_unit:int, maximum_layers:int):
    '''
    return a list of tuple of neural networks
    '''
    neural_network_sizes = []
    for number_of_layers in range(1,1+maximum_layers):
        units = int(np.power(min( size*percent, maximum_unit ), 1/number_of_layers))
        neural_network_sizes.append(tuple([units]*number_of_layers))
    return neural_network_sizes


def function_param_grid_neural_regression( K:int, bias:int,  num: int , base: int):
    start = -K
    end = K + bias 
    alpha_grid = np.logspace(start, end, num, base=base) 

    return {
        'mlp_regression__alpha' : alpha_grid
    }

