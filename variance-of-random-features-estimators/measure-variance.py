from time import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeClassifier

from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from get_data import get_data
import  utils.name_of_pipeline as name_pipeline
from sklearn.model_selection import GridSearchCV
from RFFEstimator import RFFEstimator
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mean_squared_error
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
    database , max_iter = 'w3a', 500
else:
    database , max_iter = 'a5a', 1000#300

repetitions = 5

cv = 5
n_jobs = 1

#seeds = [111,123]
seeds = [111, 222, 333,444, 555] 
#seeds = [111, 222, 333, 444, 555, 123, 345] 
seed = 123

X_train, y_train, X_test, y_test = get_data(database)
n_samples, n_features = X_train.shape


class KernelRidgeClassifier(KernelRidge):
    def predict(self, X):
        prediction = super().predict(X)
        return np.sign(prediction) 

# param grid
gamma_grid =  np.logspace(start=-5, stop=10, num = 10, base=4) ## amplica esta tranformación /n_features
alpha_grid = np.logspace(-4, 2, 6, base=10)

gamma_name =  f'{name_pipeline.kernel_ridge_classification}__{name_pipeline.gamma}'
alpha_name = f'{name_pipeline.kernel_ridge_classification}__{name_pipeline.alpha}'
param_grid = {
    gamma_name : gamma_grid / n_features,
    alpha_name: alpha_grid
}
print('Number of iterations ', max_iter)
print('Param grid:', param_grid)
        
# main method 

score_in_test = []
parameter_gamma = []
parameter_alpha = []
all_time = []
training_time_pipeline = []
for seed in [1]:

    pipeline_kernel_method = Pipeline([
        (name_pipeline.scaler, StandardScaler()),
        (name_pipeline.kernel_ridge_classification,
          KernelRidgeClassifier(kernel='rbf') )
        ])  
    grid_search = GridSearchCV(estimator = pipeline_kernel_method, 
                               param_grid= param_grid,
                                cv=cv, n_jobs=n_jobs, refit=True,
                                scoring=make_scorer(accuracy_score))
    
    grid_search.fit(X_train, y_train)
    print(f'Kernel SVC seed {seed}')
    print(f'Best parameters {grid_search.best_params_}')
    parameter_gamma.append(grid_search.best_params_[gamma_name])
    parameter_alpha.append(grid_search.best_params_[alpha_name])
    score = grid_search.score(X_test, y_test)
    print('Score ', score)
    score_in_test.append(score)
    training_time_pipeline.append(grid_search.refit_time_)
    print('time refit', grid_search.refit_time_)

mean_score = np.mean(score_in_test)
std_score = np.std(score_in_test)
alpha_mode = stats.mode(parameter_alpha)[0][0]
gamma_mode = stats.mode(parameter_gamma)[0][0]
print('refit time ', np.mean(training_time_pipeline) )

print(f'mean_score {mean_score}+-{std_score} alpha_mode {alpha_mode} gamma_mode {gamma_mode}')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
## Procedemos a medir los métodos de componentes aleatorias 
n_components_list = np.array(
    np.round(np.linspace(start=n_samples*0.05,
                           stop= n_samples*0.40,
                           num = 40)), # a 30 salen mu bien las gráficas
                           int)

       
random_features_functions = [RBFSampler, RFFEstimator]
random_features_functions_names = ['RFF-sklearn', 'RFF-low-variance-estimator']
results = {}
#mediciones_names 
fit_time_random_features = 'fit-time-random-feature'
transform_time_random_features = 'transform-time-random-feature'
estimating_kernel = 'estimating-kernel'
training_time = 'training-time'
prediction_time = 'prediction-time'
score_in_test = 'score-in-test'
n_components_name = 'n-components'

mediciones = [fit_time_random_features, 
              score_in_test, transform_time_random_features]

training_models_names = ['linear-ridge-classification']

# Calculamos predición real 

real_kernel_matrix = rbf_kernel(X_scaled, X_scaled)

np.append(n_components_list, n_samples)
for rf in random_features_functions_names:
    results[rf] = {}
    results[rf][n_components_name] = [ c   for _ in seeds for c in n_components_list]
    for m in mediciones:
        results[rf][m] = []

for seed in seeds:

    for n_component in n_components_list:
        for rf, random_feature_model_name in zip(random_features_functions, random_features_functions_names):
            feature_model = rf(gamma=gamma_mode, n_components = n_component, random_state = seed)
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

            # error 
            mse = mean_squared_error(real_kernel_matrix, X_transformed @ X_transformed.T)
            
            results[random_feature_model_name][score_in_test].append(mse)


nombre_archivo = f'./results/{database}/results.dict'

# Abre el archivo en modo de escritura
with open(nombre_archivo, 'w') as archivo:
     archivo.write(str(results))
# Create a DataFrame
result = {}
for random_feature_model in results.keys():
    # Guardamos el dataset entero
    df = pd.DataFrame(results[random_feature_model])
    df.to_csv(f'./results/{database}/{random_feature_model}-verbose.csv', index=False)

    df_stats = pd.DataFrame(df)
    mean_results = df.groupby(n_components_name).mean()
    std_results = df.groupby(n_components_name).std()


    result['fit-time-mean'] = mean_results[fit_time_random_features]
    result['fit-time-std'] = std_results[fit_time_random_features]

    result[ 'transform-time-mean'] = mean_results[transform_time_random_features]
    result['transform-time-std'] = std_results[transform_time_random_features]


    result[ 'score-mean'] = mean_results[ score_in_test]
    result['score-std'] = std_results[ score_in_test]

    result = pd.DataFrame(result)
    result.to_csv(f'./results/{database}/{random_feature_model}-estimator-variance-kernel-stats-a.csv', index=True)

