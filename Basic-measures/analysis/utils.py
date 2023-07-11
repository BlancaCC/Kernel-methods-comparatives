def extract_parameters(file_path):
    '''
    To extract ksvm information: 

    Example: 
    >>> file_path = 'path/to/your/file.txt'
    >>> best_parameters, training_time, accuracy = extract_parameters(file_path)
    >>> print("Best Parameters:", best_parameters)
    >>> print("Training Time:", training_time)
    >>> print("Accuracy in Test:", accuracy)
    '''
    with open(file_path, 'r') as file:
        content = file.read()

    results = []
    parameters = ["Best Parameters:", "Training Time:", 'Accuracy in test:']
    for p in parameters:
        start_index = content.find(p) + len(p)
        end_index = content.find("\n", start_index)
        params_str = content[start_index:end_index]
        params = params_str#eval(params_str)
        results.append(params)

    return results

if __name__ == '__main__':

    FILES = ['/Users/blancacanocamarero/repositorios/TFM/Kernel-methods-comparatives/Basic-measures/results/accuracy_time_stats/KSVM_classification_Diabetes_cv_5.txt',
             '/Users/blancacanocamarero/repositorios/TFM/Kernel-methods-comparatives/Basic-measures/results/accuracy_time_stats/KSVM_classification_a9a_cv_5.txt',
             '/Users/blancacanocamarero/repositorios/TFM/Kernel-methods-comparatives/Basic-measures/results/accuracy_time_stats/KSVM_classification_covtype.binary_cv_2.txt'
             ]
    for file_path in FILES:
        print(file_path)
        best_parameters, training_time, accuracy = extract_parameters(file_path)
        print("Best Parameters:", best_parameters)
        print("Training Time:", training_time)
        print("Accuracy in Test:", accuracy)
        print('')