###################################################################
# Template n component function
# TODO: HACE FALTA TENER UN CONTROL DE QUÉ ES REGRESIÓN Y QUÉ CLASIFICACIÓN
###################################################################
from hyperparameters_config.param_grid_values import get_n_components_list
from hyperparameters_config.param_grid_values import percent as list_of_percents
import numpy as np
import pandas as pd
# https://scikit-learn.org/stable/modules/grid_search.html#successive-halving-user-guide
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
import joblib

from sklearn.metrics import r2_score, accuracy_score, make_scorer


from config import path_for_dataset
from utils.nested_cross_validation import nested_cross_validation
from utils.structure import data_structure, regression_type


def template_n_components(X: np.ndarray, y: np.array,
                            dataset_name:str, cv:int, n_jobs:int, model:str,
                            param_grid:dict, get_inner_estimator, without_features = False)-> tuple:
    ''' Compute nested validation cv for a dataset 

    Params
    ---------
    `X`:  Features of the data set (2d array of numpy)
    `y`: labels of the data set  (1d array of numpy)
    `dataset_name`: name of the data set, should be the name of the folder in Data's folder and should be registered in `structure.py`
    `cv`: number of folds
    `n_jobs`: number of thread in order to be parallelised 
    `model`: name of the model we are using, this is informative 
    `param_grid`: param grid (params to search the best in cross validation)
    `get_inner_estimator`:  Function with the n_components as params that return the inner_estimator created by a convenient pipeline and Transformation

    Return
    ----------
    Two dictionaries which contains array of number of components and percent. 
    For each computed percent compute its cross validation parameters and score.
    `cv_results_total`: save each cv results
    `results_total`: save the statistic of all cv
    ```
    cv_results_total = { 
        'percent': [],
        "n_components":[],
        "Score in test":[],
        "Training Time":[],
        "Best Parameters": [],
        "Best Score in CV": [], 
    }
    results_total = {
        'percent' : [],
        'n_components' :[],
        "Mean Score in test" :[],
        "Std Score in test" :[],
        "Mean Training Time": [],
        "Std Training Time": [],
        "Mean Best Score in CV": [],
        "Std Best Score in CV": [] 
    }
    ```

    Example of params grid:
    ---------------------------- 
    >>> n_rows,dim = X.shape
    >>> l_alpha = np.array([10.**k for k in range(-3, 5, 1)])
    >>> l_gamma = np.logspace(-4, 10,6, base=2)/ dim
    >>> param_grid ={
    >>>             'regressor__fourier_random_features__gamma': l_gamma,
    >>>             'regressor__ridge__alpha': l_alpha}
    >>> # Create the pipeline
    >>>     pipeline = Pipeline([
    >>>         ('scaler', StandardScaler()),
    >>>         ('fourier_random_features', RBFSampler(n_components= n_components)),
    >>>         ('ridge', Ridge())
    >>>     ])    
    >>>    inner_estimator = TransformedTargetRegressor(regressor=pipeline,
    >>>                                             transformer=StandardScaler())
   
    '''
     
    file_model_name = model.replace(" ", "_")
    file_model_name_arg = f'{file_model_name}_{dataset_name}_cv_{cv}'
    output_file_accuracy, output_file_verbose = path_for_dataset(dataset_name) 
    output_file_verbose += file_model_name_arg 
    output_file_accuracy += file_model_name_arg + '.csv'

    # Hyperparameters for CV
    hyperparameters_to_test = np.prod(list(map(lambda x: len(x), param_grid.values())))
    n_rows= X.shape[0]

    if without_features: 
        n_components_list = [n_rows] 
        percent = [100]
    else:
        n_components_list = get_n_components_list(n_rows)
        percent = list_of_percents

    # score function 
    if(data_structure[dataset_name]['type'] == regression_type):
        score_function = r2_score
    else:
        score_function = accuracy_score

    # print information
    head_title = f'''
    {'-'*20}
    'Model: {model} 
    \tDataset: {dataset_name} ({data_structure[dataset_name]['type']}) of shape {X.shape} \tCV: {cv} \tn_jobs: {n_jobs}
    \t params: { param_grid} 
    \t number of hyperparameter {hyperparameters_to_test}
    '''
    if without_features: 
        head_title += f'''
        All de dataset is used (not component selection)
        '''
    else:
        head_title += f'''
        \t n components {n_components_list}
        \t which correspond with the {percent} % of the data
        {'-'*20}
        '''
    print(head_title)
    cv_results_total = { 
        'percent': [],
        "n_components":[],
        "Score in test":[],
        "Training Time":[],
        "Best Parameters": [],
        "Best Score in CV": [], 
    }
    results_total = {
        'percent' : [],
        'n_components' :[],
        "Mean Score in test" :[],
        "Std Score in test" :[],
        "Mean Training Time": [],
        "Std Training Time": [],
        "Mean Best Score in CV": [],
        "Std Best Score in CV": [] 
    }

    for p,n_components in zip(percent,n_components_list):
        print('-'*20+f'\nNumber of components for this iteration: {n_components}')

        results_total['n_components'].append(n_components)
        cv_results_total['n_components'] += [n_components for _ in range(4)] # una por cada nested cross validation

        results_total['percent'].append(p)
        cv_results_total['percent'] += [p for _ in range(4)] 

        # Define the parameter grid
        # Create the GridSearchCV object
        inner_estimator = get_inner_estimator(n_components)
        # HalvingGridSearchCV is a faster way of implement GridSearchCV
        # why use HalvingGridSearch instead of GridSearch:
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_successive_halving_heatmap.html
        grid_search = GridSearchCV(inner_estimator, param_grid, cv=cv, n_jobs=n_jobs, refit=True, scoring=make_scorer(score_function))
        #grid_search = HalvingGridSearchCV(inner_estimator, param_grid, cv=cv, n_jobs=n_jobs, refit=True)
        # Nested cross validation
        results, cv_results = nested_cross_validation(X,y, grid_search, 4, score_function=score_function)
        for key in results.keys():
            results_total[key] += results[key]
        for key in cv_results.keys():
            cv_results_total[key] += cv_results[key]

        print(results)
        print(cv_results)

    print('Final results')
    print(results_total)
    print(cv_results_total)

    results_df = pd.DataFrame(results_total)
    results_df.to_csv(output_file_accuracy, index=False)
    results_cv_df = pd.DataFrame(cv_results_total)
    results_cv_df.to_csv(output_file_verbose+ '.csv', index=True)

    file = open(output_file_verbose+ '.txt', "w")
    file.write(str(head_title) + "\n")
    file.write(str(results_total) + "\n")
    file.write(str(cv_results_total) + "\n")
    file.close()
    # This return have no sense
    return results_df, results_cv_df