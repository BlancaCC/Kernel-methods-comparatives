# Path to save Statistic results 
path = './results/accuracy_time_stats/' 
path_for_joblib = './results/joblib/'
accuracy = 'accuracy_time_stats/'
verbose = 'verboses/'
joblib = 'joblibs/'


def path_for_dataset(ds):
    path = f'./results/{ds}/'
    return path + accuracy, path + verbose