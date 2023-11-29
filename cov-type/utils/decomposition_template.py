try:
        from measure_average_time import measure_average_time
        from mean_average_result import get_iterations_mean
except:
        from utils.measure_average_time import measure_average_time
        from utils.mean_average_result import get_iterations_mean

def decomposition_train_method(results_dict, model, 
                               transformed_X, y, X_test, y_test, fitted_scaler, 
                               data_transformer,  name_transformation,  model_name):
        # Training time 
        results_dict[f'{name_transformation}-{model_name}-training-average-time'], results_dict[f'{name_transformation}-{model_name}-training-std-time'], _ = measure_average_time(model.fit, (transformed_X, y))

        # n_iterations
        results_dict[f'{name_transformation}-{model_name}-n_iterations_in_training_mean'], results_dict[f'{name_transformation}-{model_name}-n_iterations_in_training_std'] = get_iterations_mean(transformed_X, y, model)

        # Time in predict average
        model.fit(transformed_X, y)
        results_dict[f'{name_transformation}-{model_name}-predict-average-time'], results_dict[f'{name_transformation}-{model_name}-predict-std-time'], _ = measure_average_time(model.predict, [transformed_X])

        # score 
        transformed_X_test = data_transformer.transform(fitted_scaler.transform(X_test))
        results_dict[f'{name_transformation}-{model_name}-score-average-time'], results_dict[f'{name_transformation}-{model_name}-score-std-time'], results_dict[f'{name_transformation}-{model_name}-score-in-test'] = measure_average_time(
                model.score, (transformed_X_test, y_test))
        
        results_dict['method'] = f'{name_transformation}-{model_name}'
        return results_dict
