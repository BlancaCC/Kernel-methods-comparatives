from time import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.kernel_approximation import Nystroem, RBFSampler

from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from get_data import get_data
import  utils.name_of_pipeline as name_pipeline
from sklearn.model_selection import GridSearchCV

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import argparse
parser = argparse.ArgumentParser(description="Experimentos dataset.")

# Agrega un argumento para el nombre del dataset
parser.add_argument("dataset_name", help="Nombre del dataset a procesar")

# Analiza los argumentos de la línea de comandos
args = parser.parse_args()

if args.dataset_name == 'Diabetes':
    database, max_iter = 'Diabetes', 10000
elif args.dataset_name == 'w3a':
    database , max_iter = 'w3a', 350#400
else:
    database , max_iter = 'a5a', 2000

repetitions = 2

cv = 3
n_jobs = 5

seeds = [111]
seeds = [111, 222, 333,444] 
seeds = [111, 222, 333, 444, 555, 123, 345] 
#seeds = [111, 222] 
seed = 123


X_train, y_train, X_test, y_test = get_data(database)
n_samples, n_features = X_train.shape


tolerance = 10**(-10)
#alpha_grid = np.logspace(-4, 2, 6, base=10)

# param grid
gamma_grid =  np.logspace(start=-5, stop=10, num = 8, base=4) ## amplica esta tranformación /n_features
c_grid = np.logspace(-4, 4, 7, base=10)


gamma_name =  f'{name_pipeline.kernel_svm}__{name_pipeline.gamma}'
c_name = f'{name_pipeline.kernel_svm}__{name_pipeline.c}'
param_grid = {
    gamma_name : gamma_grid / n_features,
    c_name: c_grid 
}

# a5a best score
#param_grid = {'kernel_svm__C': [1.0], 'kernel_svm__gamma': [0.05890885621228663]}
print('Number of iterations ', max_iter)
print('Param grid:', param_grid)
        
# main method 

score_in_test = []
parameter_gamma = []
parameter_c = []
all_time = []
training_time_pipeline = []
for seed in seeds:

    pipeline_kernel_method = Pipeline([
        (name_pipeline.scaler, StandardScaler()),
        (name_pipeline.kernel_svm, SVC(kernel='rbf',
        max_iter=max_iter, 
        tol=tolerance, 
        random_state= seed) )
        ])  
    grid_search = GridSearchCV(estimator = pipeline_kernel_method, 
                               param_grid= param_grid,
                                cv=cv, n_jobs=n_jobs, refit=True,
                                scoring=make_scorer(accuracy_score))
    
    grid_search.fit(X_train, y_train)
    print(f'Kernel SVC seed {seed}')
    print(f'Best parameters {grid_search.best_params_}')
    parameter_gamma.append(grid_search.best_params_[gamma_name])
    parameter_c.append(grid_search.best_params_[c_name])
    score = grid_search.score(X_test, y_test)
    print('Score ', score)
    score_in_test.append(score)
    training_time_pipeline.append(grid_search.refit_time_)

# TODO: guardar tiempo de refit y compararlo
mean_score = np.mean(score_in_test)
std_score = np.std(score_in_test)
c_mode = stats.mode(parameter_c)[0][0]
gamma_mode = stats.mode(parameter_gamma)[0][0]
print('param grid',param_grid)
print('refit time ', np.mean(training_time_pipeline) )

print(f'mean_score {mean_score}+-{std_score} c_mode {c_mode} gamma_mode {gamma_mode}')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
## Procedemos a medir los métodos de componentes aleatorias 
n_components_list = np.array(
    np.round(np.linspace(start=5,
                           stop= n_samples*0.15,
                           num = 30)),
                           int)

n_small = np.array(np.round(np.linspace(start=5,
                           stop= n_features*1.5,
                           num = 0)),
                           int)
n_components_list = [*n_small, *n_components_list]

print(n_components_list)
random_features_functions = [RBFSampler, Nystroem]
random_features_functions_names = ['RFF', 'Nystroem']
results = {}
#mediciones_names 
fit_time_random_features = 'fit-time-random-feature'
transform_time_random_features = 'transform-time-random-feature'
training_time = 'training-time'
prediction_time = 'prediction-time'
score_in_test = 'score-in-test'
n_components_name = 'n-components'

mediciones = [fit_time_random_features, 
              transform_time_random_features]

training_models_names = ['SVC-kernel-linear', 
                         #'linear-SVC-dual-true', 
                         'linear-SVC-dual-false'
                        ]

np.append(n_components_list, n_samples)
for rf in random_features_functions_names:
    results[rf] = {}
    results[rf][n_components_name] = [ c   for _ in seeds for c in n_components_list]
    for m in mediciones:
        results[rf][m] = []
    for tm in training_models_names:
        results[rf][f'{training_time}-{tm}'] = []
        results[rf][f'{prediction_time}-{tm}'] = []
        results[rf][f'{score_in_test}-{tm}'] = []
  

for seed in seeds:
    training_models = [
    SVC(C= c_mode, kernel='linear', 
       max_iter=max_iter, tol=tolerance, random_state=seed),
    #LinearSVC( loss = 'hinge',
    #    C=c_mode, random_state=seed,
    #    max_iter=max_iter, tol=tolerance,
    #    dual=True),
    LinearSVC( 
        C=c_mode, random_state=seed,
        max_iter=max_iter, loss='squared_hinge', penalty='l2', dual=False)
]
    
    for n_component in n_components_list:
        for rf, random_feature_model_name in zip(random_features_functions, random_features_functions_names):
            feature_model = rf(gamma=gamma_mode, n_components = n_component, 
                       random_state=seed)
            #print(f'random feature model {random_feature_model_name} n_components {n_component} seed {seed}  database {database}')

            # fit random features
            start = time()
            feature_model.fit(X_scaled)
            end = time()
            results[random_feature_model_name][fit_time_random_features].append(end - start)

            # Transform random features 
            start = time()
            X_transformed = feature_model.transform(X_scaled)
            end = time()
            results[random_feature_model_name][transform_time_random_features].append(end - start)

            for training_model, training_model_name in zip(training_models, training_models_names):
                # Training models
                start = time()
                training_model.fit(X_transformed, y_train)
                end = time()
                results[random_feature_model_name][f'{training_time}-{training_model_name}'].append(end - start)
                
                # Prediction time in test date 
                start = time()
                X_test_transformed = feature_model.transform(X_test_scaled)
                y_prediction = training_model.predict(X_test_transformed)
                end = time()
                results[random_feature_model_name][f'{prediction_time}-{training_model_name}'].append(end - start)
                
                # score 
                results[random_feature_model_name][f'{score_in_test}-{training_model_name}'].append(accuracy_score(y_test, y_prediction))
                print(f'Score {random_feature_model_name} {training_model_name} component {n_component}: | {accuracy_score(y_test, y_prediction)} |')

nombre_archivo = f'./results/{database}/results.dict'

# Abre el archivo en modo de escritura
with open(nombre_archivo, 'w') as archivo:
     archivo.write(str(results))
# Create a DataFrame
for random_feature_model in results.keys():
    # Guardamos el dataset entero
    df = pd.DataFrame(results[random_feature_model])
    df.to_csv(f'./results/{database}/{random_feature_model}-verbose.csv', index=False)

    df_stats = pd.DataFrame(df)
    mean_results = df.groupby(n_components_name).mean()
    std_results = df.groupby(n_components_name).std()


    for training_model in training_models_names:
        result = pd.DataFrame()
        #result[n_components_name] = n_components_list
       
        result['fit-time-mean'] = mean_results['fit-time-random-feature']
        result['fit-time-std'] = std_results['fit-time-random-feature']

        result[ 'transform-time-mean'] = mean_results['fit-time-random-feature']
        result['transform-time-std'] = std_results['transform-time-random-feature']

        result[ 'training-time-mean'] = mean_results[ f'training-time-{training_model}']
        result['training-time-std'] = std_results[ f'training-time-{training_model}']

        result[ 'prediction-time-mean'] = mean_results[ f'prediction-time-{training_model}']
        result['prediction-time-std'] = std_results[ f'prediction-time-{training_model}']

        result[ 'score-mean'] = mean_results[ f'score-in-test-{training_model}']
        result['score-std'] = std_results[ f'score-in-test-{training_model}']

        result.to_csv(f'./results/{database}/{random_feature_model}-{training_model}-stats-a.csv', index=True)
    


### ------- Kernel method ---------------
results = {
    training_time : [],
    prediction_time:[],
    score_in_test: []
}
training_model_name = 'kernel-SVC'
for seed in seeds:
    print(training_model_name+' '+str(seed))
    '''
    training_model = SVC(kernel='rbf',
            max_iter=max_iter, 
            tol=tolerance, 
            random_state= seed,
            gamma=gamma_mode, C=c_mode)
    '''
    training_model =  SVC(C= c_mode, kernel='linear', 
       max_iter=max_iter, tol=tolerance, random_state=seed)
    # Porque estamos con el método total
    feature_model = Nystroem(gamma=gamma_mode, n_components = n_samples, 
                       random_state=seed)
    # fit random features
    start = time()
    X_transformed = feature_model.fit_transform(X_scaled)
    training_model.fit(X_transformed, y_train)
    end = time()
    results[training_time].append(end - start)

    # Prediction time in test date 
    X_test_transformed = feature_model.transform(scaler.transform(X_test))
    start = time()
    y_prediction = training_model.predict(X_test_transformed)
    end = time()
    results[prediction_time].append(end - start)
    
    # score 
    results[score_in_test].append(accuracy_score(y_test, y_prediction))
    


results_df = pd.DataFrame(results)
results_df.to_csv(f'./results/{database}/{training_model_name}-verbose.csv', index=False)

# Crear un DataFrame con las estadísticas
stats_df = pd.DataFrame()

# Calcular las medias y desviaciones estándar para cada columna
stats_df['mean-refit-time-mean'] = [np.mean(training_time_pipeline)]
stats_df['mean-refit-time-std'] = [np.mean(training_time_pipeline)]
for column in results.keys():
    stats_df[f'{column}-mean'] = [np.mean(results[column])]
    stats_df[f'{column}-std'] = [np.std(results[column])]
    print(column, np.mean(results[column]))
print(stats_df)

stats_df.to_csv(f'./results/{database}/{training_model_name}-stats.csv', index=False)

