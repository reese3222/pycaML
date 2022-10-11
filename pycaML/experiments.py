import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pickle
from os.path import exists
from sklearn import metrics
from sklearn.model_selection import cross_validate
from datetime import datetime
from statistics import mean
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from sklearn import model_selection
from .models import models_class, models_reg, models_stacking_class, models_stacking_reg
from .models import bagging_models, boosting_models, simple_models_reg, simple_models_class


class Experiment():
    result = None
    names = []
    params = []
    spaces = []
    params = []
    trials = []
    parameters = []
    models_opt = []
    results_def = []
    stack_estimators = []

    # use all cores except one
    n_j = -2

    def __init__(self, name, stacking=False, tuning=False, array=True):
        # name: name of the experiment
        # stacking: True or False

        self.tuning = tuning
        self.stacking = stacking
        self.name = name

        dirs = ['params', 'trials', 'tables', 'data']
        for i in dirs:
            path = f'experiments/{self.name}/{i}'
            if not exists(path):
                os.makedirs(path)
                print(f'Directory {path} created!')

        if tuning is True and stacking is True:
            self.exp_type = 'stacking_tuning'
        elif tuning is True and stacking is False:
            self.exp_type = 'tuning'
        elif tuning is False and stacking is True:
            self.exp_type = 'stacking'
        else:
            self.exp_type = 'default'

        file = f'experiments/{self.name}/tables/{self.name}_{self.exp_type}.csv'
        if exists(file):
            self.result = pd.read_csv(file)
            print('Result loaded')

        for i in self.models.keys():
            # function that loads parameters from file
            file = f'experiments/{self.name}/params/{self.name}_{i}.pkl'
            if exists(file):
                with open(file, 'rb') as f:
                    self.models[i]['opt_params'] = pickle.load(f)
                    # print(f'Parameters for {i} loaded')
                    # print(self.models[i]['opt_params'] )

        self.dataset = [
                        f'experiments/{self.name}/data/X_train.csv',
                        f'experiments/{self.name}/data/y_train.csv',
                        f'experiments/{self.name}/data/X_test.csv',
                        f'experiments/{self.name}/data/y_test.csv'
                        ]

        if all(exists(i) for i in self.dataset):
            self.X_train = pd.read_csv(self.dataset[0]).to_numpy() if array else pd.read_csv(self.dataset[0])
            self.y_train = pd.read_csv(self.dataset[1]).to_numpy().ravel() if array else pd.read_csv(self.dataset[1])
            self.X_test = pd.read_csv(self.dataset[2]).to_numpy() if array else pd.read_csv(self.dataset[2])
            self.y_test = pd.read_csv(self.dataset[3]).to_numpy().ravel() if array else pd.read_csv(self.dataset[3])
            print('Data loaded')

    def load_data(self, path, target, split=0.2, array=True):
        data = pd.read_csv(path)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.drop(target, axis=1),
            data[target],
            test_size=split,
            random_state=42
        )

        self.X_train.to_csv(self.dataset[0], index=False)
        self.y_train.to_csv(self.dataset[1], index=False)
        self.X_test.to_csv(self.dataset[2], index=False)
        self.y_test.to_csv(self.dataset[3], index=False)
        print('X_train, y_train, X_test, y_test saved')

        if array:
            self.X_train = self.X_train.to_numpy()
            self.X_test = self.X_test.to_numpy()
            self.y_train = self.y_train.to_numpy().ravel()
            self.y_test = self.y_test.to_numpy().ravel()

    # trains the models and saves the results
    def start(self,  cv=5, raise_errors=False, n_eval=100):
        if self.result is not None:
            return
        if isinstance(self, RegressionExperiment):
            self.scoring = ['neg_mean_squared_error']
            mean_mse = []
            std_mse = []
            mse_test = []
            duration_time = []
        else:
            mean_F1 = []
            std_F1 = []
            F1_test = []
            duration_time = []
            precision = []
            recall = []
            self.n_classes = len(np.unique(self.y_train))
            self.scoring = ['f1_weighted'] if self.n_classes > 2 else ['f1', 'precision', 'recall']

        if self.stacking:
            self.build_stack()
            self.models = self.stacking_models.copy()

        for (k, model_name) in enumerate(self.models.keys()):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            if self.stacking is True:
                model = self.models[model_name]

            if self.exp_type == 'tuning':
                model = self.models[model_name]
                print(current_time, ': 📘 Optimizing', model_name, '📘')

                file = f'experiments/{self.name}/params/{self.name}_{model_name}.pkl'
                if exists(file):
                    print('📘', model_name, 'already tuned 📘')
                    with open(file, 'rb') as f:
                        params = pickle.load(f)
                else:
                    params = self.optimize_model(model_name=model_name, n_eval=n_eval)
                    with open(file, 'wb+') as f:
                        pickle.dump(params, f)
                    print(file, 'saved')

                self.models[model_name]['opt_params'] = params.copy()
                # print('Optimized parameters', params)
                model = self.models[model_name]['algo'](**self.models[model_name]['opt_params'])

            else:
                model = self.models[model_name]['algo'](**self.models[model_name]['def_params'])

            print(f'{current_time}: Training {model_name} {k+1}/{len(self.models)+1}          ', end='\r')
            scores = cross_validate(model,
                                    self.X_train,
                                    self.y_train,
                                    scoring=self.scoring,
                                    cv=cv,
                                    n_jobs=self.n_j,
                                    error_score="raise"
                                    )

            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            self.y_pred = y_pred

            if isinstance(self, RegressionExperiment):
                mean_mse.append(np.sqrt(-mean(scores['test_neg_mean_squared_error'])))
                mse_test.append(np.sqrt(metrics.mean_squared_error(self.y_test, y_pred)))
                duration_time.append(mean(scores['fit_time']))
                std_mse.append(np.std(np.sqrt(-scores['test_neg_mean_squared_error'])))
            else:
                if self.n_classes > 2:
                    mean_F1.append(mean(scores['test_f1_weighted']))
                    std_F1.append(np.std(scores['test_f1_weighted']))
                    F1_test.append(metrics.f1_score(self.y_test, y_pred, average='weighted'))
                    duration_time.append(mean(scores['fit_time']))
                else:
                    mean_F1.append(mean(scores['test_f1']))
                    std_F1.append(np.std(scores['test_f1']))
                    F1_test.append(metrics.f1_score(self.y_test, y_pred))
                    duration_time.append(mean(scores['fit_time']))
                    precision.append(metrics.precision_score(self.y_test, y_pred))
                    recall.append(metrics.recall_score(self.y_test, y_pred))
        # append to the results dataframe the results of the model

        if isinstance(self, RegressionExperiment):
            result = {
                'Model': self.models.keys(),
                'RMSE_test': mse_test,
                'RMSE_cv': mean_mse,
                'RMSE_std': std_mse,
                'Time': duration_time,
            }

            result = pd.DataFrame(result).sort_values(by='RMSE_test', ascending=True)

        else:
            result = {
                'Model': self.models.keys(),
                'F1_test': F1_test,
                'F1_cv': mean_F1,
                'F1_Std': std_F1,
                'Time': duration_time,
            }

            if self.n_classes == 2:
                result['Precision_test'] = precision
                result['Recall_test'] = recall

            result = pd.DataFrame(result).sort_values(by='F1_test', ascending=False)

        self.result = result
        result.to_csv(f'experiments/{self.name}/tables/{self.name}_{self.exp_type}.csv', index=False)

    def optimize_model(self, model_name, n_eval):
        def objective(space, model=self.models[model_name]['algo']):
            model = model(**space)
            losses = model_selection.cross_val_score(
                model,
                self.X_train,
                self.y_train,
                scoring=self.scoring[0],
                n_jobs=self.n_j)

            return {'status': STATUS_OK,
                    'loss': -mean(losses),
                    'loss_variance': np.var(losses, ddof=1)}

        trial = Trials()
        best = fmin(objective,
                    space=self.models[model_name]['space'],
                    algo=tpe.suggest,
                    max_evals=n_eval,
                    trials=trial,
                    show_progressbar=True
                    )
        file = f'experiments/{self.name}/trials/trial_{self.name}_{model_name}.pkl'
        with open(file, 'wb+') as f:
            pickle.dump(trial, f)
        return space_eval(self.models[model_name]['space'], best)

    # method used for passing the parameters to the model.
    # Stacking diverse models takes the best model from boosting and the best model from bagging
    def build_stack(self):
        if self.tuning:
            base_result = pd.read_csv(f'experiments/{self.name}/tables/{self.name}_tuning.csv')
            estimators_stacking_all = [(i, j['algo'](**self.models[i]['opt_params'])) for i, j in self.models.items()]
        else:
            base_result = pd.read_csv(f'experiments/{self.name}/tables/{self.name}_default.csv')
            estimators_stacking_all = [(i, j['algo'](**self.models[i]['def_params'])) for i, j in self.models.items()]

        best_boosting = base_result.loc[base_result['Model'].isin(boosting_models)].iloc[0, 0]
        best_bagging = base_result.loc[base_result['Model'].isin(bagging_models)].iloc[0, 0]
        self.stacking_estimators_names_diverse.append(best_bagging)
        self.stacking_estimators_names_diverse.append(best_boosting)

        self.stacking_models['Stacking (all)']['def_params']['estimators'] = estimators_stacking_all
        voting_estimators = [i for i in estimators_stacking_all if i[0] != 'Ridge Classifier']
        self.stacking_models['Voting (all)']['def_params']['estimators'] = voting_estimators
        self.stacking_models_diverse = {m: self.models[m] for m in self.stacking_estimators_names_diverse}

        stack_dict = self.stacking_models_diverse.items()
        if self.tuning is False:
            estimators_stacking_diverse = [(i, j['algo'](**self.models[i]['def_params'])) for i, j in stack_dict]
        else:
            estimators_stacking_diverse = [(i, j['algo'](**self.models[i]['opt_params'])) for i, j in stack_dict]

        self.stacking_models['Stacking (diverse)']['def_params']['estimators'] = estimators_stacking_diverse
        self.stacking_models['Voting (diverse)']['def_params']['estimators'] = estimators_stacking_diverse


class RegressionExperiment(Experiment):
    def __init__(self, name, tuning=False, stacking=False):
        self.stacking_models = models_stacking_reg
        self.models = models_reg
        self.scoring = ['neg_mean_squared_error']
        self.stacking_estimators_names_diverse = simple_models_reg
        super().__init__(name=name, tuning=tuning, stacking=stacking)


class ClassificationExperiment(Experiment):
    def __init__(self, name, tuning=False, stacking=False):
        self.models = models_class
        self.stacking_models = models_stacking_class
        self.stacking_estimators_names_diverse = simple_models_class
        super().__init__(name=name, tuning=tuning, stacking=stacking)
