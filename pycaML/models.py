"""
This module is used to store the models  and search spaces used in pycaML.
"""

from sklearn import linear_model
from hyperopt import hp
import numpy as np
from lightgbm import LGBMRegressor
from hyperopt.pyll import scope
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import Lars
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import HuberRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

space_en = {
    'random_state': 322,
    'alpha': hp.uniform('alpha', 0, 10),
    'l1_ratio': hp.uniform('l1_ratio', 0, 1),
    'max_iter': scope.int(hp.quniform('max_iter', 200, 5000, 100))
            }

space_lasso = {
    'random_state': 322,
    'alpha': hp.uniform('alpha', 0, 10),
    'max_iter': scope.int(hp.quniform('max_iter', 200, 5000, 100))
            }

space_ridge = {
    'random_state': 322,
    'alpha': hp.uniform('alpha', 0, 10),
    'max_iter': scope.int(hp.quniform('max_iter', 200, 5000, 100))
            }

space_et = {
    'random_state': 322,
    'n_jobs': -2,
    'n_estimators': scope.int(hp.quniform('n_estimators', 20, 300, 10, )),
    'max_depth':        hp.choice('max_depth', [None, scope.int(hp.uniform('max_depth2', 1, 8))]),
    'min_samples_split': hp.choice('min_samples_split', [2, hp.uniform('min_samples_split2', 0, 1)]),
    'min_samples_leaf': hp.choice('min_samples_leaf', [1, hp.uniform('min_samples_leaf2', 0, 1)]),
    'max_features':        hp.choice('max_features', ['sqrt', 'log2', None])
    }

space_rf = space_et.copy()

space_tree = {
    'random_state': 322,
    'max_depth':        hp.choice('max_depth', [None, scope.int(hp.uniform('max_depth2', 1, 8))]),
    'min_samples_split': hp.choice('min_samples_split', [2, hp.uniform('min_samples_split2', 0, 1)]),
    'min_samples_leaf': hp.choice('min_samples_leaf', [1, hp.uniform('min_samples_leaf2', 0, 1)])
    }

space_xgb = {
    'random_state': 322,
    'n_estimators':        scope.int(hp.quniform('n_estimators', 20, 250, 10)),
    'eta': hp.loguniform('eta', -7, 0),
    'max_depth': scope.int(hp.uniform('max_depth', 1, 8)),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
    'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
    'alpha': hp.choice('alpha', [0, hp.loguniform('alpha_positive', -16, 2)]),
    'lambda': hp.choice('lambda', [0, hp.loguniform('lambda_positive', -16, 2)]),
    'gamma': hp.choice('gamma', [0, hp.loguniform('gamma_positive', -16, 2)]),
    # 'tree_method': hp.choice('tree_method', ['gpu_hist'])
}

space_lgbm = {
    'random_state': 322,
    'verbose': -1,
    'n_estimators':        scope.int(hp.quniform('n_estimators', 20, 250, 10)),
    'learning_rate': hp.loguniform('learning_rate', -7, 0),
    'min_data_in_leaf': scope.int(hp.loguniform('min_data_in_leaf', 0.7, 7)),
    'max_depth':        hp.choice('max_depth', [-1, scope.int(hp.uniform('max_depth2', 1, 8))]),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
    'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
    'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
}

space_adab = {
    'random_state': 322,
    'n_estimators':     scope.int(hp.quniform('n_estimators', 20, 250, 10)),
    'learning_rate':    hp.loguniform('learning_rate', -7, 0.7),
}

space_gb = {
    'random_state': 322,
    'max_iter':     scope.int(hp.quniform('max_iter', 20, 250, 10)),
    'learning_rate':    hp.loguniform('learning_rate', -5.0, -0.7),
    'max_depth':        scope.int(hp.uniform('max_depth', 1, 8)),
    'max_leaf_nodes': scope.int(hp.loguniform('max_leaf_nodes', 0.7, 7)),
    'min_samples_leaf': scope.int(hp.loguniform('min_data_in_leaf', 0.7, 7)),
    'l2_regularization': hp.loguniform('l2_regularization', -10, -0.01)
    }

space_ebm = {
    'random_state': 322,
    'max_rounds':     scope.int(hp.quniform('max_rounds', 200, 7000, 100)),
    'learning_rate':    hp.loguniform('learning_rate', -6.0, -0.7),
    'interactions':        scope.int(hp.uniform('interactions', 1, 50)),
    'min_samples_leaf': scope.int(hp.loguniform('min_samples_leaf', 0.7, 7)),
    'max_leaves': scope.int(hp.loguniform('max_leaves', 0.7, 7)),
    }

space_LOGReg = {
    'random_state': 322,
    'C': hp.uniform('C', 0, 1000),
    'max_iter': 10000
    }

space_cb = {
        'silent': hp.choice('silent', [True]),
        'random_seed': 322,
        'thread_count': 2,
        # 'task_type': hp.choice('task_type',['GPU']),
        'depth':        scope.int(hp.uniform('depth', 1, 8)),
        'learning_rate': hp.loguniform('learning_rate', -5, 0),
        'random_strength': hp.choice('random_strength', [1, 20]),
        'l2_leaf_reg': hp.loguniform('l2_leaf_reg', 0, np.log(10)),
        'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
        'used_ram_limit': '30gb',
        'min_data_in_leaf': scope.int(hp.loguniform('min_data_in_leaf', 0, 7))
    }

space_lda = {
    'solver': hp.choice('solver', ['svd', 'lsqr', 'eigen']),
    'store_covariance': hp.choice('store_covariance', [True, False]),
    'tol': hp.loguniform('tol', -10, -1)
    }

space_gnb = {
    'var_smoothing': hp.loguniform('var_smoothing', -10, -1)
    }

space_qda = {
    'reg_param': hp.loguniform('reg_param', -10, -1)
    }

space_par = {
    'C': hp.loguniform('C', -7, 7),
    'fit_intercept': hp.choice('fit_intercept', [True, False]),
    'max_iter': scope.int(hp.uniform('max_iter', 1, 1000)),
    'tol': hp.loguniform('tol', -10, -1),
    'early_stopping': hp.choice('early_stopping', [True, False]),
    'validation_fraction': hp.uniform('validation_fraction', 0, 1),
    'n_iter_no_change': scope.int(hp.uniform('n_iter_no_change', 1, 100)),
    'shuffle': hp.choice('shuffle', [True, False]),
    'epsilon': hp.uniform('epsilon', 0, 1),
    'random_state': 322,
    'warm_start': hp.choice('warm_start', [True, False]),
    'average': hp.choice('average', [True, False])
    }

space_svm = {
    'C': hp.loguniform('C', -7, 7),
    'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
    'degree': scope.int(hp.uniform('degree', 1, 6)),
    'gamma': hp.choice('gamma', ['scale', 'auto']),
    'coef0': hp.loguniform('coef0', -7, 7),
    'shrinking': hp.choice('shrinking', [True, False]),
    'tol': hp.loguniform('tol', -10, -1),
    'cache_size': hp.loguniform('cache_size', 1, 4),
    'max_iter': scope.int(hp.uniform('max_iter', -1, 10000)),

    }

space_ransac = {
    'base_estimator': hp.choice('base_estimator', [
        None,
        LinearRegression(),
        LogisticRegression(),
        Ridge(),
        Lasso(),
        ElasticNet()]),
    'min_samples': scope.int(hp.uniform('min_samples', 1, 100)),
    'max_trials': scope.int(hp.uniform('max_trials', 1, 100)),
    'residual_threshold': hp.loguniform('residual_threshold', -7, 7),
    'max_skips': scope.int(hp.uniform('max_skips', 1, 100)),
    'stop_n_inliers': scope.int(hp.uniform('stop_n_inliers', 1, 100)),
    'stop_probability': hp.uniform('stop_probability', 0, 1),
    'stop_score': hp.uniform('stop_score', 0, 1),
    'random_state': 322
    }

space_pac = {
    'C': hp.loguniform('C', -7, 7),
    'fit_intercept': hp.choice('fit_intercept', [True, False]),
    'max_iter': scope.int(hp.uniform('max_iter', 1, 10000)),
    'tol': hp.loguniform('tol', -10, -1),
    'early_stopping': hp.choice('early_stopping', [True, False]),
    'n_iter_no_change': scope.int(hp.uniform('n_iter_no_change', 1, 10000)),
    'shuffle': hp.choice('shuffle', [True, False]),
    'verbose': hp.choice('verbose', [False]),
    'loss': hp.choice('loss', ['hinge', 'squared_hinge']),
    'random_state': 322,
    'average': hp.choice('average', [False, True])
    }

space_omp = {
    'n_nonzero_coefs': scope.int(hp.uniform('n_nonzero_coefs', 1, 100)),
    'tol': hp.loguniform('tol', -10, -1),
    'precompute': hp.choice('precompute', ['auto', True, False]),
    'normalize': hp.choice('normalize', [False]),
    'fit_intercept': hp.choice('fit_intercept', [True, False]),
    }

space_knn = {
    'n_neighbors': scope.int(hp.uniform('n_neighbors', 1, 100)),
    'weights': hp.choice('weights', ['uniform', 'distance']),
    'leaf_size': scope.int(hp.uniform('leaf_size', 1, 100)),
    'p': scope.int(hp.uniform('p', 1, 10)),
    }

space_lars = {
    'fit_intercept': hp.choice('fit_intercept', [True, False]),
    'normalize': hp.choice('normalize', [False]),
    'n_nonzero_coefs': scope.int(hp.uniform('n_nonzero_coefs', 1, 100)),
    'eps': hp.loguniform('eps', -10, -1),
    'random_state': 322,
    }

space_svc = {
    'C': hp.loguniform('C', -6, 6),
    'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
    'degree': scope.int(hp.uniform('degree', 1, 10)),
    'gamma': hp.choice('gamma', ['scale', 'auto']),
    'coef0': hp.loguniform('coef0', -7, 7),
    'shrinking': hp.choice('shrinking', [True, False]),
    'probability': hp.choice('probability', [True]),
    'tol': hp.loguniform('tol', -10, -1),
    'cache_size': hp.loguniform('cache_size', 1, 4),
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'max_iter': scope.int(hp.uniform('max_iter', -1, 10000)),
    'decision_function_shape': hp.choice('decision_function_shape', ['ovr']),
    # 'break_ties': hp.choice('break_ties', [False]),
    'random_state': 322
    }

space_perceptron = {
    'penalty': hp.choice('penalty', [None, 'l2', 'l1', 'elasticnet']),
    'alpha': hp.loguniform('alpha', -8, 6),
    'max_iter': scope.int(hp.uniform('max_iter', 100, 10000)),
    'tol': hp.loguniform('tol', -10, -1),
    'eta0': hp.loguniform('eta0', -7, 7),
    'random_state': 322,
    'validation_fraction': hp.uniform('validation_fraction', 0, 1),
    'n_iter_no_change': scope.int(hp.uniform('n_iter_no_change', 1, 10000)),
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    }

space_mlp = {
    'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(100,), (100, 100), (100, 100, 100)]),
    'activation': hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu']),
    'alpha': hp.loguniform('alpha', -7, 6),
    'batch_size': hp.choice('batch_size', ['auto', 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]),
    'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive']),
    'learning_rate_init': hp.loguniform('learning_rate_init', -7, 7),
    'power_t': hp.uniform('power_t', 0, 1),
    'max_iter': scope.int(hp.uniform('max_iter', 100, 10000)),
    'random_state': 322,
    'tol': hp.loguniform('tol', -10, -1),
    'momentum': hp.uniform('momentum', 0, 1),
    'beta_1': hp.uniform('beta_1', 0, 1),
    'beta_2': hp.uniform('beta_2', 0, 1),
    'epsilon': hp.loguniform('epsilon', -10, -1),
    }

space_sgd = {
    'penalty': hp.choice('penalty', [None, 'l2', 'l1', 'elasticnet']),
    'alpha': hp.loguniform('alpha', -8, 6),
    'l1_ratio': hp.uniform('l1_ratio', 0, 1),
    'max_iter': scope.int(hp.uniform('max_iter', 100, 10000)),
    'tol': hp.loguniform('tol', -10, -1),
    'shuffle': hp.choice('shuffle', [True, False]),
    'epsilon': hp.loguniform('epsilon', -10, -1),
    'learning_rate': hp.choice('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']),
    'eta0': hp.loguniform('eta0', -7, 7),
    'power_t': hp.uniform('power_t', 0, 1),
    'random_state': 322
    }

space_ransac = {
    'max_trials': scope.int(hp.uniform('max_trials', 1, 100)),
    'stop_n_inliers': scope.int(hp.uniform('stop_n_inliers', 1, 100)),
    'stop_probability': hp.uniform('stop_probability', 0, 1),
    'stop_score': hp.uniform('stop_score', 0, 1),
    'residual_threshold': hp.uniform('residual_threshold', 0, 1),
    'random_state': 322
    }

space_br = {
    'n_iter': scope.int(hp.uniform('n_iter', 100, 10000)),
    'tol': hp.loguniform('tol', -8, -1),
    'alpha_1': hp.loguniform('alpha_1', -8, -1),
    'alpha_2': hp.loguniform('alpha_2', -8, -1),
    'lambda_1': hp.loguniform('lambda_1', -8, -1),
    'lambda_2': hp.loguniform('lambda_2', -8, -1),
    'compute_score': hp.choice('compute_score', [False, True]),
    'fit_intercept': hp.choice('fit_intercept', [True, False]),
    'copy_X': hp.choice('copy_X', [True]),
    'verbose': hp.choice('verbose', [False]),
    }


space_bagging_reg = {
    'base_estimator': hp.choice('base_estimator', [
        None,
        LinearRegression(),
        Ridge(),
        Lasso(),
        ElasticNet(),
        Lars(),
        BayesianRidge(),
        ARDRegression(),
        SGDRegressor(),
        PassiveAggressiveRegressor(),
        HuberRegressor(),
        TheilSenRegressor(),
        SVR(),
        MLPRegressor()]),
    'n_estimators': scope.int(hp.uniform('n_estimators', 1, 100)),
    'max_samples': hp.uniform('max_samples', 0.3, 1),
    'max_features': hp.uniform('max_features', 0, 1),
    'bootstrap': hp.choice('bootstrap', [True, False]),
    'bootstrap_features': hp.choice('bootstrap_features', [True, False]),
    'random_state': 322,
    }

space_bagging_class = {
    'base_estimator': hp.choice('base_estimator', [
        None,
        LogisticRegression(),
        RidgeClassifier(),
        Perceptron(),
        SGDClassifier(),
        PassiveAggressiveClassifier(),
        Perceptron(),
        PassiveAggressiveClassifier(),
        MLPClassifier(),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        GaussianNB()
    ]),
    'n_estimators': scope.int(hp.uniform('n_estimators', 1, 100)),
    'max_samples': hp.uniform('max_samples', 0.3, 1),
    'max_features': hp.uniform('max_features', 0, 1),
    'bootstrap': hp.choice('bootstrap', [True, False]),
    'bootstrap_features': hp.choice('bootstrap_features', [True, False]),
    'random_state': 322,
}

space_huber = {
    'epsilon': hp.uniform('epsilon', 1.1, 1.35),
    'max_iter': scope.int(hp.uniform('max_iter', 100, 10000)),
    'alpha': hp.loguniform('alpha', -7, 7),
    'tol': hp.loguniform('tol', -10, -1),
    'warm_start': hp.choice('warm_start', [True, False]),
    'fit_intercept': hp.choice('fit_intercept', [True, False])
    }

space_theil = {
    'max_iter': scope.int(hp.uniform('max_iter', 100, 10000)),
    'tol': hp.loguniform('tol', -10, -1),
    'random_state': 322
    }

models_stacking_reg = {
        'Stacking': StackingRegressor(estimators = []),
        'Voting': VotingRegressor(estimators = []),
}

models_stacking_class = {
        'Stacking': StackingClassifier(estimators = []),
        'Voting (hard)': VotingClassifier(estimators = [], voting = 'hard'),
        'Voting (soft)': VotingClassifier(estimators = [], voting = 'soft'),
}


models_reg = {
        'Bagging': {
            'algo': BaggingRegressor,
            'space': space_bagging_reg,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Decision Tree': {
            'algo': DecisionTreeRegressor,
            'space': space_tree,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Random Forest': {
            'algo': RandomForestRegressor,
            'space': space_rf,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'XGBoost': {
            'algo': XGBRegressor,
            'space': space_xgb,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'LightGBM': {
            'algo': LGBMRegressor,
            'space': space_lgbm,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Gradient Boost': {
            'algo': HistGradientBoostingRegressor,
            'space': space_gb,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'CatBoost': {
            'algo': CatBoostRegressor,
            'space': space_cb,
            'opt_params': {},
            'def_params': {'silent': True, 'random_seed': 322}
            },
        'ExtraTrees': {
            'algo': ExtraTreesRegressor,
            'space': space_et,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'KNN': {
            'algo': KNeighborsRegressor,
            'space': space_knn,
            'opt_params': {},
            'def_params': {}
            },
        'AdaBoost': {
            'algo': AdaBoostRegressor,
            'space': space_adab,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        # 'EBM': {
        #     'algo': ExplainableBoostingRegressor,
        #     'space': space_ebm,
        #     'opt_params': {},
        #     'def_params': {'random_state': 322}
        #     },
        'Elastic Net': {
            'algo': ElasticNet,
            'space': space_en,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Lasso': {
            'algo': Lasso,
            'space': space_lasso,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Ridge': {
            'algo': Ridge,
            'space': space_ridge,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Support Vector Machine': {
            'algo': SVR,
            'space': space_svm,
            'opt_params': {},
            'def_params': {},
            },
        'Least Angle Regression': {
            'algo': Lars,
            'space': space_lars,
            'opt_params': {},
            'def_params': {'normalize': False, 'random_state': 322}
            },
        'Orthogonal Matching Pursuit': {
            'algo': OrthogonalMatchingPursuit,
            'space': space_omp,
            'opt_params': {},
            'def_params': {'normalize': False},
            },
        'Multi-layer Perceptron': {
            'algo': MLPRegressor,
            'space': space_mlp,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Passive Aggressive': {
            'algo': PassiveAggressiveRegressor,
            'space': space_par,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Bayesian Ridge': {
            'algo': BayesianRidge,
            'space': space_br,
            'opt_params': {},
            'def_params': {}
            },
        'Huber': {
            'algo': HuberRegressor,
            'space': space_huber,
            'opt_params': {},
            'def_params': {}
            },
        'TheilSen': {
            'algo': TheilSenRegressor,
            'space': space_theil,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'RANSAC': {
            'algo': RANSACRegressor,
            'space': space_ransac,
            'opt_params': {},
            'def_params': {'random_state': 322}
            }
        
        }

models_class = {
        'Bagging': {
            'algo': BaggingClassifier,
            'space': space_bagging_class,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Decision Tree': {
            'algo': DecisionTreeClassifier,
            'space': space_tree,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Linear Discriminant Analysis': {
            'algo': LinearDiscriminantAnalysis,
            'space': space_lda,
            'opt_params': {},
            'def_params': {}
            },
        'Gaussian Naive Bayes': {
            'algo': GaussianNB,
            'space': space_gnb,
            'opt_params': {},
            'def_params': {}
            },
        'Support Vector Machine': {
            'algo': SVC,
            'space': space_svc,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Quadratic Discriminant Analysis': {
            'algo': QuadraticDiscriminantAnalysis,
            'space': space_qda,
            'opt_params': {},
            'def_params': {}
            },
        'Passive Aggressive Classifier': {
            'algo': PassiveAggressiveClassifier,
            'space': space_pac,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Random Forest': {
            'algo': RandomForestClassifier,
            'space': space_rf,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'XGBoost': {
            'algo': XGBClassifier,
            'space': space_xgb,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'LightGBM': {
            'algo': LGBMClassifier,
            'space': space_lgbm,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Gradient Boost': {
            'algo': HistGradientBoostingClassifier,
            'space': space_gb,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'CatBoost': {
            'algo': CatBoostClassifier,
            'space': space_cb,
            'opt_params': {},
            'def_params': {'silent': True, 'random_seed': 322}
            },
        'ExtraTrees': {
            'algo': ExtraTreesClassifier,
            'space': space_et,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'KNN': {
            'algo': KNeighborsClassifier,
            'space': space_knn,
            'opt_params': {},
            'def_params': {}
            },
        'AdaBoost': {
            'algo': AdaBoostClassifier,
            'space': space_adab,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Logistic Regression': {
            'algo': LogisticRegression,
            'space': space_LOGReg,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Ridge Classifier': {
            'algo': RidgeClassifier,
            'space': space_ridge,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Perceptron': {
            'algo': Perceptron,
            'space': space_perceptron,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Multi-layer Perceptron': {
            'algo': MLPClassifier,
            'space': space_mlp,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
    }
