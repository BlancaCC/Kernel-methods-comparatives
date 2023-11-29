from numpy import mean, std, array

def average_result(func, args, repetitions=5):
    measures = []

    for _ in range(repetitions):

        result = func(*args)
        if result == None: 
               return None, None
        measures.append(result)
    try:
        measures = array(list(map(lambda x: x[0], measures)))
    except:
           pass
    
    mean_result = mean(measures)
    standard_deviation = std(measures)

    return mean_result, standard_deviation


# particular situation: 
def get_iterations(X, y, classifier):
            classifier.fit(X,y)
            try: 
                result = classifier.n_iter_
            except:
                result = None
            return result 


def get_iterations_mean(X, y, classifier):
        func = get_iterations
        args = (X, y, classifier)
        return average_result(func, args, repetitions=5)
       