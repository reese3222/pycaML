# this file is used for storing search spaces and models

from sklearn import linear_model
from hyperopt import hp
import numpy as np
from lightgbm import LGBMRegressor
from hyperopt.pyll import scope
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.glassbox import ExplainableBoostingRegressor

bagging_models = ['Random Forest', 'ExtraTrees']
boosting_models = ['EBM', 'XGBoost', 'CatBoost', 'LightGBM', 'Gradient Boost']
simple_models_reg = ['KNN', 'Linear Regression', 'Decision Tree']
simple_models_class = ['KNN', 'Logistic Regression', 'Decision Tree']

space_en = {
    'random_state': 322,
    # 'alpha': hp.uniform('alpha', 0,100),
    'l1_ratio': hp.uniform('l1_ratio', 0, 1),
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
    'tree_method': hp.choice('tree_method', ['gpu_hist'])
}

space_lgbm = {
    'random_state': 322,
    'n_estimators':        scope.int(hp.quniform('n_estimators', 20, 250, 10)),
    'learning_rate': hp.loguniform('learning_rate', -7, 0),
    'min_data_in_leaf': scope.int(hp.loguniform('min_data_in_leaf', 0.7, 7)),
    'max_depth':        hp.choice('max_depth', [-1, scope.int(hp.uniform('max_depth2', 1, 8))]),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
    'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
    'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
    'device': hp.choice('device', ['gpu'])
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

space_knn = {
    'n_neighbors':     scope.int(hp.uniform('n_neighbors', 1, 100)),
    # 'metric': hp.choice('metric',['l1','l2','manhattan','cosine', 'haversine']),
    'n_jobs': -2
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

models_class = {
        'Decision Tree': {
            'algo': DecisionTreeClassifier,
            'space': space_tree,
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
            'def_params': {'tree_method': 'gpu_hist', 'random_state': 322}
            },
        'LightGBM': {
            'algo': LGBMClassifier,
            'space': space_lgbm,
            'opt_params': {},
            'def_params': {'device': 'gpu', 'random_state': 322}
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
        'EBM': {
            'algo': ExplainableBoostingClassifier,
            'space': space_ebm,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Logistic Regression': {
            'algo': LogisticRegression,
            'space': space_LOGReg,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
    }


models_stacking_reg = {
        'Stacking (all)': {
            'algo': StackingRegressor,
            'def_params': {'final_estimator': linear_model.LinearRegression()},
            },
        'Voting (all)': {
            'algo': VotingRegressor,
            'def_params': {},
            },

        'Stacking (diverse)': {
            'algo': StackingRegressor,
            'def_params': {'final_estimator': linear_model.LinearRegression()},
            },
        'Voting (diverse)': {
            'algo': VotingRegressor,
            'def_params': {},
            },
}

models_stacking_class = {
        'Stacking (all)': {
            'algo': StackingClassifier,
            'def_params': {'final_estimator': linear_model.LogisticRegression()},
            },
        'Voting (all)': {
            'algo': VotingClassifier,
            'def_params': {'voting': 'soft'},
            },
        'Stacking (diverse)': {
            'algo': StackingClassifier,
            'def_params': {'final_estimator': linear_model.LogisticRegression()},
            },
        'Voting (diverse)': {
            'algo': VotingClassifier,
            'def_params': {'voting': 'soft'},
            },
}


models_reg = {
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
            'def_params': {'tree_method': 'gpu_hist', 'random_state': 322}
            },
        'LightGBM': {
            'algo': LGBMRegressor,
            'space': space_lgbm,
            'opt_params': {},
            'def_params': {'device': 'gpu', 'random_state': 322}
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
        'EBM': {
            'algo': ExplainableBoostingRegressor,
            'space': space_ebm,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
        'Linear Regression': {
            'algo': ElasticNet,
            'space': space_en,
            'opt_params': {},
            'def_params': {'random_state': 322}
            },
    }
