######################################################################
# Some criterium in orden to choose hyperparameters
# C and gamma KSVM (18 y 19 de las notas se discute cómo gestionar C y gamma en clasificación)
# 25 el épsilon en regresión
# https://github.com/joseDorronsoro/SVMs-for-Classification-and-Regression/blob/master/notes_svm_classification_regression_2021.pdf 

import numpy as np

######################################################################
#                        For  KSVM             
######################################################################

def function_param_grid_ksvm(dimension:int, K:int, bias: int, base: int):
    start = -K
    end = K + bias 
    num = 1
    gamma_space = np.logspace(start, end, num, base=base) / dimension

    return {
        'svm__C' : np.logspace(-2, 4, num, base=10),
        'svm__gamma': gamma_space,
        
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
        'ridge_classification__alpha' : np.logspace(-4, 1, 5, base=10)
    }


######################################################################
#                        For  RBF + Ridge Classification              
######################################################################

def function_param_grid_rbf_ridge_classification(dimension:int, K:int, bias: int, base: int):
    start = -K
    end = K + bias 
    num = 5
    gamma_space = np.logspace(start, end, num, base=base) / dimension

    return {
        'rbf_sampler__gamma' : gamma_space,
        'ridge_classification__alpha' : np.logspace(-4, 2, 5, base=10)
    }