import argparse
from get_data import get_data_without_split, get_data
from utils.structure import  data_structure
# Models to test 
# Ridge  classification family
from models_classification.kernel_ridge_classification import nested_kernel_ridge_classification
from models_classification.random_Fourier_features_ridge_classification import nested_random_Fourier_features_ridge_classification
from models_classification.Nystrom_ridge_classification import nested_Nystrom_ridge_classification
# SVC family 
from models_classification.ksvm import nested_kernel_svm
from models_classification.Nystrom_SVC import nested_Nystrom_SVC
from models_classification.random_Fourier_features_SVC import nested_random_Fourier_features_SVC

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Model training and evaluation')
    parser.add_argument('dataset', type=str, help='Name of the dataset')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs (default: 1)')
    parser.add_argument('--cv', type=int, default=5, help='Number of CV splits (default: 5)')
    args = parser.parse_args()

    X,y, X_test, y_test = [False for _ in range(4)]
    # Get data 
    if data_structure[args.dataset]['has_test'] == True:
        X,y, X_test, y_test = get_data(args.dataset)
    else:
        X,y = get_data_without_split(args.dataset)
    # Models 
    # Ridge Regressions family
    nested_kernel_ridge_classification(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs, X_test=X_test, y_test = y_test)
    nested_random_Fourier_features_ridge_classification(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs, X_test=X_test, y_test = y_test)
    nested_Nystrom_ridge_classification(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs, X_test=X_test, y_test = y_test)
    # SVC family
    nested_kernel_svm(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs, X_test=X_test, y_test = y_test)
    nested_Nystrom_SVC(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs, X_test=X_test, y_test = y_test)
    nested_random_Fourier_features_SVC(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs, X_test=X_test, y_test = y_test)