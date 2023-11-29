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
# Train shape:  (768, 8) 
# Test shape: - 

#  w3a 
# Type of problem: classification 
# Train shape:  (4912, 300) 
# Test shape:  (44837, 300)

#  a5a 
# Type of problem: classification 
# Train shape:  (6414, 123) 
# Test shape:  (26147, 123)

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
# Train shape:  (581012, 54) 
# Test shape: - 

#  eunite2001 
# Type of problem: regression 
# Train shape:  (336, 16) 
# Test shape:  (31, 16)

#  abalone 
# Type of problem: regression 
# Train shape:  (4177, 8) 
# Test shape: - 

#  CPU_SMALL 
# Type of problem: regression 
# Train shape:  (8192, 12) 
# Test shape: - 
#################################################
# 

classification_type = 'classification'
regression_type = 'regression'

data_structure = {
    'Diabetes' : {
        'train' : 'diabetes',
        'test' : 0.3,
        'has_test':False, # poner a false
        'n_features': 8,
        'type': classification_type,
    },
    'w3a' : {
        'train' : 'w3a',
        'test' : 'w3a.t',
        'has_test':True,
        'n_features': 300,
        'type': classification_type,
    },
    'a5a' : {
        'train' : 'a5a',
        'test' : 'a5a.t',
        'has_test':True,
        'n_features': 123,
        'type': classification_type,
    },
    'a7a' : {
        'train' : 'a7a',
        'test' : 'a7a.t',
        'has_test':True,
        'n_features': 123,
        'type': classification_type,
    },
    'a9a' : {
        'train' : 'a9a.txt',
        'test' : 'a9a.t',
        'has_test':True,
        'n_features': 123,
        'type': classification_type,
    },
    'covtype.binary' :{
        'train' : 'covtype.libsvm.binary',
        'test' : 0.3, 
        'has_test':False,
        'n_features' : 54,
        'type': classification_type,
    },
     'eunite2001' : {
        'train' : 'eunite2001',
        'test' : 'eunite2001.t',
        'has_test': True,
        'n_features': 16,
        'type': regression_type,
    },
    'abalone' : {
        'train' : 'abalone',
        'test' : 0.8,
        'has_test': False, 
        'n_features': 8,
        'type': regression_type,
    },
    'CPU_SMALL' : {
        'train' : 'cpusmall',
        'test' : 0.3,
        'has_test':False,
        'n_features': 12,
        'type': regression_type,
        'has_test':False
    } 
}