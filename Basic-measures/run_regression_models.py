import argparse
from get_data import get_data_without_split

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

    # Get data 
    X,y = get_data_without_split(args.dataset)
    # Models 
    # Ridge Regressions family
    nested_kernel_ridge_regression(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs)
    nested_random_Fourier_features_ridge_regression(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs)
    nested_Nystrom_ridge_regression(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs)
    # SVC family
    nested_kernel_SVR(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs)
    nested_Nystrom_SVR(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs)
    nested_random_Fourier_features_SVR(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs)