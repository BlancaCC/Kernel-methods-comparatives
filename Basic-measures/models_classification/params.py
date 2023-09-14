######################################################################
# Some criterium in orden to choose hyperparameters
# C and gamma KSVM (18 y 19 de las notas se discute cómo gestionar C y gamma en clasificación)
# 25 el épsilon en regresión
# https://github.com/joseDorronsoro/SVMs-for-Classification-and-Regression/blob/master/notes_svm_classification_regression_2021.pdf 

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
        'svm__C' : np.logspace(-2, 4, 7, base=10), # [0.1, 1, 10, 100, 1000]
        'svm__gamma': gamma_space,
        
    }


######################################################################
#                        For  Kernel ridge classification     
######################################################################

def function_param_grid_kernel_ridge_classification(dimension:int, K:int, bias: int, base: int, num = 6):
    start = -K
    end = K + bias 
    gamma_space = np.logspace(start, end, num, base=base) / dimension

    return {
        f'{name_pipeline.kernel_ridge_classification}__gamma' : gamma_space,
        f'{name_pipeline.kernel_ridge_classification}__alpha' : np.logspace(-4, 2, num, base=10)
      }

# Number from random features
n_components_list = [10, 20, 50, 100, 200, 500, 1000]


######################################################################
#                        For  Nyström + Ridge Classification              
######################################################################

def function_param_grid_nystrom_ridge_classification(dimension:int, K:int, bias: int, base: int):
    start = -K
    end = K + bias 
    num = 5
    gamma_space = np.logspace(start, end, num, base=base) / dimension

    return {
        'nystroem__gamma' : gamma_space,
        'ridge_classification__alpha' : np.logspace(-4, 2, 5, base=10)
      }


######################################################################
#                        For  Nyström + SVM Classification              
######################################################################

def function_param_grid_nystrom_svm_classification(dimension:int, K:int, bias: int, base: int,  num:int):
    start = -K
    end = K + bias 
    gamma_space = np.logspace(start, end, num, base=base) / dimension
    return {
        'nystroem__gamma' : gamma_space,
        'svm__C' : np.logspace(-4, 2, num, base=10)
    }



######################################################################
#                        For  RBF + Ridge Classification              
######################################################################

def function_param_grid_rbf_ridge_classification(dimension:int, K:int, bias: int, base: int, num = 5):
    start = -K
    end = K + bias 
    gamma_space = np.logspace(start, end, num, base=base) / dimension
    return {
        'rbf_sampler__gamma' : ['scale'] + list(gamma_space),
        'ridge_classification__alpha' : np.logspace(-4, 2, num, base=10)
    }



######################################################################
#                        For  rbf + SVM Classification              
######################################################################

def function_param_grid_rbf_svm_classification(dimension:int, K:int, bias: int, base: int,  num:int):
    start = -K
    end = K + bias 
    gamma_space = np.logspace(start, end, num, base=base) / dimension
    return {
        'rbf_sampler__gamma' : ['scale'] + list(gamma_space),
        'svm__C' : np.logspace(-4, 2, num, base=10)
    }


#############################################################################
#                   Neural Network Classification 
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


def function_param_grid_neural_classification( K:int, bias:int,  num: int , base: int):
    start = -K
    end = K + bias 
    alpha_grid = np.logspace(start, end, num, base=base) 

    return {
        'mlp_classification__alpha' : alpha_grid
    }

