alpha = 'alpha'
gamma = 'gamma'
C = 'C'
svm = 'svm'
ridge = 'ridge'


best_hyperparameters = {
    'Diabetes' : {
        # "{'kernel_ridge_classification__alpha': 0.0001, 'kernel_ridge_classification__gamma': 0.0008501470344688711}"
        ridge : {
            alpha :  0.0001,
            gamma: 0.0008501470344688711,
        },
         # 'kernel_svm__C': 10.0, 'kernel_svm__gamma': 0.0013810679320049757
        svm: {
            gamma: 0.0013810679320049757,
            C: 10,
        }
    },
    'w3a' : {
        #{'kernel_ridge_classification__alpha': 0.3981071705534969, 'kernel_ridge_classification__gamma': 3.2552083333333335e-06}"
         ridge : {
            alpha :  0.3981071705534969,
            gamma: 3.2552083333333335e-06,
        },
        # {'kernel_svm__C': 0.01, 'kernel_svm__gamma': 3.2552083333333335e-06}
        svm: {
            gamma: 3.2552083333333335e-06,
            C: 0.01,
        }
    },
    'a5a' : {
        # {'kernel_ridge_classification__alpha': 0.001584893192461114, 'kernel_ridge_classification__gamma': 7.939532520325204e-06}
        ridge : {
            alpha :   0.001584893192461114,
            gamma: 7.939532520325204e-06,
        },
        # {'kernel_svm__C': 1000.0, 'kernel_svm__gamma': 7.939532520325204e-06}
        svm: {
            gamma: 7.939532520325204e-06,
            C: 1000.0,
        }
    },
     'eunite2001' : {
         # {'regressor__kernel_ridge__alpha': 0.025118864315095794, 'regressor__kernel_ridge__gamma': 0.0029603839189656206}"
         ridge : {
            alpha :  0.025118864315095794,
            gamma: 0.0029603839189656206,
        },
        # 'regressor__kernel_svm__C': 1000.0, 'regressor__kernel_svm__gamma': 6.103515625e-05
        svm: {
            gamma: 6.103515625e-05,
            C: 1000.0,
        }
    },
    'abalone' : {
        # {'regressor__kernel_ridge__alpha': 0.025118864315095794, 'regressor__kernel_ridge__gamma': 0.0412346222116529}
        ridge : {
            alpha :  0.025118864315095794,
            gamma:  0.0412346222116529,
        },
        # "{'regressor__kernel_svm__C': 1.0, 'regressor__kernel_svm__gamma': 0.1767766952966369}
        svm: {
            gamma: 0.1767766952966369,
            C: 1.0,
        }
    },
    'CPU_SMALL' : {
        # {'regressor__kernel_ridge__alpha': 0.0001, 'regressor__kernel_ridge__gamma': 0.0039471785586208275}",
         ridge : {
            alpha :  0.0001,
            gamma:  0.0039471785586208275,
        },
        # {'regressor__kernel_svm__C': 100.0, 'regressor__kernel_svm__gamma': 0.010416666666666666}"
        svm: {
            gamma: 0.010416666666666666,
            C: 100,
        }
    } 
}