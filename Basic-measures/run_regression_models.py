import argparse
from get_data import get_data_without_split, get_data
from utils.structure import data_structure
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print('Using intelex')
except:
    print(f'Working with normal CPU.')

# Models to test 
# Ridge  regression family
from models_regression.kernel_ridge_regression import nested_kernel_ridge_regression
from models_regression.random_Fourier_features_ridge_regression import nested_random_Fourier_features_ridge_regression
from models_regression.Nystrom_ridge_regression import nested_Nystrom_ridge_regression
# SVR family 
from models_regression.kernel_SVR import nested_kernel_SVR
from models_regression.Nystrom_SVR import nested_Nystrom_SVR
from models_regression.random_Fourier_features_SVR import nested_random_Fourier_features_SVR
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
    nested_kernel_ridge_regression(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs, X_test=X_test, y_test = y_test)
    nested_random_Fourier_features_ridge_regression(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs, X_test=X_test, y_test = y_test)
    nested_Nystrom_ridge_regression(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs, X_test=X_test, y_test = y_test)
    # SVC family
    nested_kernel_SVR(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs, X_test=X_test, y_test = y_test)
    nested_Nystrom_SVR(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs, X_test=X_test, y_test = y_test)
    nested_random_Fourier_features_SVR(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs, X_test=X_test, y_test = y_test)