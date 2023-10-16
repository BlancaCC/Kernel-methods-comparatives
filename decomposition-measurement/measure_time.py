import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from utils.decomposition_template import decomposition_train_method
from get_data import get_data
from Data.best_hyperparameter_found import best_hyperparameters, svm, ridge, alpha, gamma, C
from hyperparameters_config.param_grid_values import get_n_components_list

from utils.measure_average_time import measure_average_time
import hyperparameters_config.name_of_pipeline as name_pipeline


from nystroem_features_generation import get_Nystroem_transformer
from random_features_generation import get_RFF_transformer

# Classification models
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

import pandas as pd


def measure_time(dataset_name, max_iter = 10**6):

    # Read dataset 
    X, y, X_test, y_test = get_data(dataset_name)
    n_rows, n_features = X.shape

    # Read best hyperparameters
    best_hyperparameter = best_hyperparameters[dataset_name]
    n_component_list = get_n_components_list(n_rows)
    
    rows_of_df = []
    seed = 123
    print(f''' 
          Measurement type of {dataset_name} of shape {X.shape}
                n_component_list = {n_component_list}
                seed = {seed}
{'-'*100}
''')
    for n_component in n_component_list:
        print(n_component)
        measure_results = dict()
        measure_results['n-component'] = n_component
        measure_results['n-rows'] = n_rows
        measure_results['n-features'] = n_features

        # Measuring transform time 
        # For classification problem
        scaler = StandardScaler()
        scaler.fit(X)

        measure_results['scaler-average-time'], measure_results['scaler-std-time'], transformed_X =measure_average_time(scaler.transform, [X])

        # Define transformers
        RFF = get_RFF_transformer(best_hyperparameter[svm][gamma],
                        n_component, seed)
        Nystroem = get_Nystroem_transformer(best_hyperparameter[svm][gamma], n_component, seed)

        # Measure fit time
        measure_results['RFF-fit-average-time'], measure_results['RFF-fit-std-time'], _ = measure_average_time(RFF.fit, [transformed_X])
        measure_results['Nystroem-fit-average-time'], measure_results['Nystroem-fit-std-time'], _ = measure_average_time(Nystroem.fit, [transformed_X])
        RFF.fit(transformed_X)

        # Transform time
        measure_results['RFF-transform-average-time'], measure_results['RFF-transform-std-time'], RFF_transformed_X = measure_average_time(RFF.transform, [transformed_X])
        measure_results['Nystroem-transform-average-time'], measure_results['Nystroem-transform-std-time'], Nystroem_transformed_X = measure_average_time(Nystroem.transform, [transformed_X])

        # Training time 
        # Classification problem 
        # ------ RIDGE CLASSIFIER -------
        print('Ridge classifier')
        ridge_classifier = RidgeClassifier(alpha= best_hyperparameter[ridge][alpha],
                                            random_state=seed, 
                                            max_iter= max_iter)
    
        measure_results = decomposition_train_method(results_dict=measure_results, 
                                                     model=ridge_classifier, 
                                                     transformed_X= RFF_transformed_X, y=y, X_test=X_test, y_test= y_test,
                                                     fitted_scaler=scaler, data_transformer= RFF,
                                                    name_transformation='RFF', model_name='ridge'
            
        )
        measure_results = decomposition_train_method(results_dict=measure_results, 
                                                     model=ridge_classifier,
                                                     transformed_X= Nystroem_transformed_X, y=y, X_test=X_test, y_test= y_test,
                                                     fitted_scaler=scaler, data_transformer=Nystroem,
                                                    name_transformation='Nystroem', model_name='ridge'
            
        )
        
        # ------ SVC CLASSIFIER -------
        print('SVC')
        svc_model = SVC(C= best_hyperparameter[svm][C],
                        kernel='linear',
                        random_state=seed, 
                        max_iter= max_iter)
        #svc_model.fit(X,y)
        #print('IMPRIMIMOS SVC', svc_model.n_iter_)
        measure_results = decomposition_train_method(results_dict=measure_results, 
                                                     model=svc_model, 
                                                     transformed_X= RFF_transformed_X, y=y, X_test=X_test, y_test= y_test,
                                                     fitted_scaler=scaler, data_transformer= RFF,
                                                    name_transformation='RFF', model_name='svc')
            
        measure_results = decomposition_train_method(results_dict=measure_results, 
                                                     model=svc_model,
                                                     transformed_X= Nystroem_transformed_X, y=y, X_test=X_test, y_test= y_test,
                                                     fitted_scaler=scaler, data_transformer=Nystroem,
                                                    name_transformation='Nystroem', model_name='svc')
            
    
        # ------ Linear SVM CLASSIFIER -------
        print('LinearSVC')
        svc_model = LinearSVC(C= best_hyperparameter[svm][C],
                        penalty='l2', 
                        loss='hinge',
                        random_state=seed, 
                        max_iter= max_iter)
        measure_results = decomposition_train_method(results_dict=measure_results, 
                                                     model=svc_model, 
                                                     transformed_X= RFF_transformed_X, y=y, X_test=X_test, y_test= y_test,
                                                     fitted_scaler=scaler, data_transformer= RFF,
                                                    name_transformation='RFF', model_name='linear-svc')
            
        
        measure_results = decomposition_train_method(results_dict=measure_results, 
                                                     model=svc_model,
                                                     transformed_X= Nystroem_transformed_X, y=y, X_test=X_test, y_test= y_test,
                                                     fitted_scaler=scaler, data_transformer=Nystroem,
                                                    name_transformation='Nystroem', model_name='linear-svc')
            
        measure_results['random-state'] = seed
        rows_of_df.append(measure_results)
    # Create a DataFrame from the results list
    df_random_features  = pd.DataFrame(rows_of_df)

    # ------  KERNEL METHODS -------
    class identity():
        def transform(X):
            return X
    print('Kernel svc')
    kernel_results = dict()
    ksvm_model = SVC(C= best_hyperparameter[svm][C],
                    kernel='rbf',
                    gamma= best_hyperparameter[svm][gamma],
                    random_state=seed, 
                    max_iter= max_iter)
    ksvm_result = decomposition_train_method(results_dict=kernel_results, 
                                                     model= ksvm_model,
                                                     transformed_X= transformed_X, y=y, X_test=X_test, y_test= y_test,
                                                     fitted_scaler=scaler, data_transformer= identity,
                                                    name_transformation='kernel', model_name='svc')
    class KernelRidgeClassifier(KernelRidge):
        n_iter_ = None
        def predict(self, X):
            prediction = super().predict(X)
            return np.sign(prediction) 
        
        def score(self, X, y, sample_weight=None):
            return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
        
    print('Kernel ridge classification')        
    kernel_ridge_model = KernelRidgeClassifier(
                    alpha== best_hyperparameter[ridge][alpha],          
                    kernel='rbf',
                    gamma= best_hyperparameter[ridge][gamma])
    kernel_ridge_result = dict()

    pipeline = Pipeline([
    (name_pipeline.scaler, StandardScaler()),
    (name_pipeline.kernel_ridge_classification, KernelRidgeClassifier(kernel='rbf', 
                                                                      alpha= best_hyperparameter[ridge][alpha],
                                                                      gamma= best_hyperparameter[ridge][gamma]) )
    ])
    pipeline.fit(X,y)
    kernel_ridge_result['kernel-ridge-score-in-test'] = pipeline.score(X_test, y_test)

    kernel_ridge_result = decomposition_train_method(results_dict=kernel_ridge_result, 
                                                     model= kernel_ridge_model,
                                                     transformed_X= transformed_X, y=y, X_test=X_test, y_test= y_test,
                                                     fitted_scaler=scaler, data_transformer= identity,
                                                    name_transformation='kernel', model_name='ridge')
    
    df_kernel = pd.DataFrame([ksvm_result, kernel_ridge_result])
    return df_random_features,  df_kernel

if __name__ == '__main__':
    import sys

    dataset_name = sys.argv[1]
    #for dataset_name in ['Diabetes', 'w3a', 'a5a']:
    df_random_features, df_kernel = measure_time(dataset_name)
    file_path = f"/Users/blancacanocamarero/repositorios/TFM/Kernel-methods-comparatives/decomposition-measurement/RESULT-DECOMPOSITION-MEASUREMENT/{dataset_name}"

    # Guarda el DataFrame en un archivo CSV
    df_random_features.to_csv(file_path+"_random_features.csv", index=False)
    df_kernel.to_csv(file_path+"_kernel_methods.csv", index=False)

    
        
