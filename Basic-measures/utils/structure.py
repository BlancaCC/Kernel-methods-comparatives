#################################################
# Data set, folder structure
# key: folder
# Train: Training file
# Test: if str then test file, if float split rate
# n_features: n_features
# Type of problem
###################################
#  Diabetes 
# Type of problem: classification 
# Train shape:  (537, 8) 
# Test shape:  (231, 8)

#  w3a 
# Type of problem: classification 
# Train shape:  (4912, 300) 
# Test shape:  (44837, 300)

#  a7a 
# Type of problem: classification 
# Train shape:  (16100, 123) 
# Test shape:  (16461, 123)

#  a9a 
# Type of problem: classification 
# Train shape:  (32561, 123) 
# Test shape:  (16281, 123)

#  covtype.binary 
# Type of problem: classification 
# Train shape:  (406708, 54) 
# Test shape:  (174304, 54)

#  abalone 
# Type of problem: regression 
# Train shape:  (2923, 8) 
# Test shape:  (1254, 8)

#  CPU_SMALL 
# Type of problem: regression 
# Train shape:  (5734, 12) 
# Test shape:  (2458, 12)
#################################################
# 

classification_type = 'classification'
regression_type = 'regression'

data_structure = {
    'Diabetes' : {
        'train' : 'diabetes',
        'test' : 0.3,
        'n_features': 8,
        'type': classification_type,
    },
    'w3a' : {
        'train' : 'w3a',
        'test' : 'w3a.t',
        'n_features': 300,
        'type': classification_type,
    },
    'a7a' : {
        'train' : 'a7a',
        'test' : 'a7a.t',
        'n_features': 123,
        'type': classification_type,
    },
    'a9a' : {
        'train' : 'a9a.txt',
        'test' : 'a9a.t',
        'n_features': 123,
        'type': classification_type,
    },
    'covtype.binary' :{
        'train' : 'covtype.libsvm.binary',
        'test' : 0.3, 
        'n_features' : 54,
        'type': classification_type,
    },
    'abalone' : {
        'train' : 'abalone',
        'test' : 0.3,
        'n_features': 8,
        'type': regression_type,
        'has_test':False
    },
    'CPU_SMALL' : {
        'train' : 'cpusmall',
        'test' : 0.3,
        'n_features': 12,
        'type': regression_type,
        'has_test':False
    } 
}