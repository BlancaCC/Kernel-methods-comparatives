import argparse
from get_data import get_data_without_split

# Models to test 
from models_classification.kernel_ridge_classification import nested_kernel_ridge_classification


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
    nested_kernel_ridge_classification(X,y, args.dataset, cv=args.cv, n_jobs=args.n_jobs)
   