import warnings
import numpy as np
# Number from random features
percent = [*list(map(int,np.linspace(start=0.1, stop=20, num=30)))]

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
        n_components_list.sort()
        return n_components_list


if __name__ == '__main__':
        print(percent)
        n_rows_cpu_small = 8192
        print(get_n_components_list(n_rows_cpu_small))
