from typing import Tuple
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
from Data.structure import data_structure


def get_data(dataset:str) -> Tuple[np.array, np.array, np.array, np.array]:
    '''Return  X_dense, y, X_test_dense, y_test from a libsvm data set 
    Options: 
    Adults: `dataset == 'a9a' or dataset == '1'`
    '''
    # Load the dataset from a libsvm file
    if dataset in data_structure:
        train_data_file = f"./Data/{dataset}/{data_structure[dataset]['train']}"
        # Test
        test_value = data_structure[dataset]['test']
        test_in_file = isinstance(test_value, str)
        if test_in_file:
            test_data_file = f"./Data/{dataset}/{data_structure[dataset]['test']}"
        n_features = data_structure[dataset]['n_features']
    else:
        raise ValueError(f'Invalid dataset value. The valid are: {data_structure.keys()}')
    
    # Load Data
    X_sparse, y = load_svmlight_file(train_data_file, n_features=n_features)
    if test_in_file:
        X_test_sparse, y_test = load_svmlight_file(test_data_file, n_features= n_features)
    else:
        X_sparse, X_test_sparse, y, y_test = train_test_split(X_sparse, y , test_size= test_value)


    # Check if the matrix is sparse
    is_sparse = sp.issparse(X_sparse)

    # Convert the data to TensorFlow tensors
    if is_sparse:
        X_dense = np.asarray(X_sparse.todense())
        X_test_dense = np.asarray(X_test_sparse.todense())
    else:
        X_dense = np.asarray(X_sparse.toarray())
        X_test_dense = np.asarray(X_test_dense.toarray())

    return X_dense, y, X_test_dense, y_test


if __name__ == '__main__':
    for data_set in data_structure.keys():
        X_dense, y, X_test_dense, y_test = get_data(data_set)
        print('\n# ', data_set, f'\n# Type of problem: {data_structure[data_set]["type"]} \n#T rain shape: ', X_dense.shape, '\n# Test shape: ',X_test_dense.shape)