#
# 
# Random Features measurement 
#

from sklearn.kernel_approximation import RBFSampler
random_state = 123

def get_RFF_transformer(gamma:float, n_component:int, random_state = random_state):
    '''According to doc, it creates the features in the fit method'''
    return RBFSampler(gamma=gamma, n_components = n_component, random_state=random_state)