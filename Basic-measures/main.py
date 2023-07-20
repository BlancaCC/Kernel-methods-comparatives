from regression_models.Fourier_random_features_ridge import Fourier_random_features_ridge_regression_KF
from regression_models.Nystrom_ridge_regression import Nystrom_ridge_regression_KF
from get_data import get_data_without_split
from regression_models.kernel_ridge import kernel_ridge_regression_KF
import argparse

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Model training and evaluation')
    parser.add_argument('dataset', type=str, help='Name of the dataset')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs (default: 1)')
    parser.add_argument('--cv', type=int, default=5, help='Number of CV splits (default: 5)')
    args = parser.parse_args()

    # Get data 
    X,y = get_data_without_split(args.dataset)
    kernel_ridge_regression_KF(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs)
    #Nystrom_ridge_regression_KF(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs)
    #Fourier_random_features_ridge_regression_KF(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs)
    
   
