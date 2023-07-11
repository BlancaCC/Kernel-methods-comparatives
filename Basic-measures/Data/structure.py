#################################################
# Data set, folder structure
# key: folder
# Train: Training file
# Test: if str then test file, if float split rate
# n_features: n_features
# Type of problem
###################################
#  a9a 
#Type of problem: classification 
#Train shape:  (32561, 123) 
#Test shape:  (16281, 123)

#  cad-rna 
#Type of problem: classification 
#Train shape:  (59535, 8) 
#Test shape:  (271617, 8)

#  covtype.binary 
#Type of problem: classification 
#Train shape:  (406708, 54) 
#Test shape:  (174304, 54)

#  CPU_SMALL 
#Type of problem: regression 
#Train shape:  (5734, 12) 
#Test shape:  (2458, 12)
#################################################

data_structure = {
    'a9a' : {
        'train' : 'a9a.txt',
        'test' : 'a9a.t',
        'n_features': 123,
        'type': 'classification'
    },
    'Diabetes' : {
        'train' : 'diabetes',
        'test' : 0.3,
        'n_features': 8,
        'type': 'classification'
    },
    'covtype.binary' :{
        'train' : 'covtype.libsvm.binary',
        'test' : 0.3, 
        'n_features' : 54,
        'type': 'classification'
    },
    'CPU_SMALL' : {
        'train' : 'cpusmall',
        'test' : 0.3,
        'n_features': 12,
        'type': 'regression'
    }
}