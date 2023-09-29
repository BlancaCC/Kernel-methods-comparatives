import warnings
# Number from random features
percent = [0.5, 1, 2,4,6,8,10,14,16,18,20,25]

def get_n_components_list(n_rows:int)->list:
        '''Gives the numbers of components 
        Params:
        `n_rows` size of the data set 
        return: 
        For the previous size return the number of components that correspond with `percents` variable[0.5, 1, 2,4,6,8,10,14,16,18,20,25]
        '''
        n_components_list = list(set(map(lambda x: int(x* n_rows / 100), percent )))
        if 0 in n_components_list:
                n_components_list = list(map(lambda x: x+1, n_components_list))
                warnings.warn('One of the components was zero', UserWarning)
        return n_components_list
