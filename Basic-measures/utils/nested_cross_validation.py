#############################################################################
##
##########################################################################
import numpy as np
from sklearn.utils import shuffle
seed = 123
# Función para dividir los datos en k pliegues
def kfold_list(X,y, k):
    '''
    Example of use: 
    -------------------- 
    >>> import numpy as np
    >>> n = 10
    >>> X = np.random.rand(n, 3)
    >>> y = np.array([i for i in range(n)])
    >>> k = 4
    >>> list(y_test for X_train, y_train, X_test, y_test in kfold_list(X,y, k))
    [array([0, 1, 2]), array([3, 4, 5]), array([6, 7]), array([8, 9])]
    '''
    X,y = shuffle(X,y, random_state=seed)
    n = len(y)
    fold_size =  n// k
    bias = n % k
    splits = []
    for i in range(k):
        start = i * fold_size + (i)*(bias >= i) + bias*(bias < i)
        end = start + fold_size + (bias > i)
  
        X_train = np.vstack((X[:start, :], X[end:, :]))
        y_train = np.array([*y[:start],*y[end:]])

        X_test = X[start:end, :]
        y_test = y[start:end]

        splits.append((X_train, y_train, X_test, y_test))
    return splits



def nested_cross_validation(X,y,grid_search, k, score_function, X_test, y_test):
    cv_results = { 
        "Best Parameters": [],
        "Best Score in CV": [],
        "Training Time":[],
        "Score in test":[],
        'cv_mean_test_score':[],
        'cv_std_test_score': [],
        'cv_mean_fit_time' : [],
        'cv_std_fit_time' : []
    }
   
    if type(X_test) is bool:
        iterate_over = kfold_list(X,y, k)
    else:
        iterate_over = [(X,y, X_test, y_test)]

    for X_train, y_train, X_test, y_test in iterate_over:
        # Fit the grid search on the data
        grid_search.fit(X_train, y_train)

        # Get the best pipeline from the grid search
        best_pipeline = grid_search.best_estimator_

        # Predict with X_test
        y_pred = best_pipeline.predict(X_test)
        # Calculate the accuracy score on the test data
        score = score_function(y_test, y_pred)
        ## Add to results
        best_index  = grid_search.best_index_
        cv_results["Best Parameters"].append(grid_search.best_params_)
        cv_results["Best Score in CV"].append(grid_search.best_score_)
        cv_results["Training Time"].append(grid_search.refit_time_)
        cv_results["Score in test"].append(score)
        cv_results['cv_mean_test_score'].append(grid_search.cv_results_['mean_test_score'][best_index])
        cv_results['cv_std_test_score'].append(grid_search.cv_results_['std_test_score'][best_index])
        cv_results['cv_mean_fit_time'].append(grid_search.cv_results_['mean_fit_time'][best_index])
        cv_results['cv_std_fit_time'].append(grid_search.cv_results_['std_fit_time'][best_index])
        if X_test == False:
            cv_results['cv_std_test_score'][-1] = 0

    results = {
        "Mean Score in test" :[np.mean(cv_results["Score in test"])],
        "Std Score in test" :[np.std(cv_results["Score in test"])],
        "Mean Best Score in CV": [np.mean(cv_results["Best Score in CV"])],
        "Std Best Score in CV": [np.std(cv_results["Best Score in CV"])],
        "Mean Training Time": [np.mean(cv_results["Training Time"])],
        "Std Training Time": [np.std(cv_results["Training Time"])],
        'cv_mean_test_score':[np.mean(cv_results['cv_mean_test_score'])],
        'cv_std_test_score': [np.mean(cv_results['cv_std_test_score'])],
        'cv_mean_fit_time':[np.mean(cv_results['cv_mean_fit_time'])],
        'cv_std_fit_time': [np.mean(cv_results['cv_std_fit_time'])],
    }
    return results, cv_results