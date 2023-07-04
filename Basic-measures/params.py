######################################################################
# Some criterium in orden to choose hyperparameters
# C and gamma KSVM (18 y 19 de las notas se discute cómo gestionar C y gamma en clasificación)
# 25 el épsilon en regresión
# https://github.com/joseDorronsoro/SVMs-for-Classification-and-Regression/blob/master/notes_svm_classification_regression_2021.pdf 

import numpy as np

######################################################################
#                        For  KSVM             
######################################################################
param_grid_ksvm = {
    'svm__C': [0.01, 0.1, 1, 10, 100, 1000], # 0.1 or 1 && 1000 or 10000
    'svm__gamma': [0.01, 0.1, 1, 1.5]
}

######################################################################
#                        For  Nyström + Ridge Classification              
######################################################################
param_grid_nystrom_ridge_classification = {
    'nystroem__gamma': [0.01, 0.1, 1.0, 1.5],
    'ridge_classification__alpha': [0.01, 0.1, 1.0, 10.0]
}
# Number of nystroem components 
n_components_list = [10, 20, 50, 100, 200, 500, 1000]


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