
from IPython.display import clear_output
import pandas as pd
import numpy as np
from hyperopt import hp
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import ignore_warnings
import os
import pickle
from os.path import exists
from sklearn import metrics
from sklearn.metrics import *
from sklearn.model_selection import cross_validate
#import convergencewarning
from sklearn.exceptions import ConvergenceWarning
from datetime import datetime
from statistics import mean
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from sklearn import model_selection
from .models import models_class, models_reg, models_stacking_class, models_stacking_reg
 

#check if model has predict_proba method


class Experiment:    

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
        """ 
        Base class for experiments. Every experiment is a folder in the experiments folder.
        It creates four subfolders: data, params, results and trials.
        The data folder contains the train and test sets.
        The params folder contains the optimized hyperparameters.
        The results folder contains the results of the experiments.
        The trials folder contains the trials of the hyperparameter optimization.

        Args:
            name (str): name of the experiment
            stacking (bool, optional): Whether it's a stacking experiment. Defaults to False.
            tuning (bool / str, optional): Specifies the metric to optimize in 
            tuning experiments. Supported values are 
            'accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'neg_mean_squared_error',
            'neg_mean_absolute_error', 'neg_median_absolute_error', 'r2'. Defaults to False.
            array (bool, optional): Whether to convert the data to numpy arrays. Defaults to True.


        Raises:
            ValueError: If tuning is not a valid metric.
        """

        if tuning not in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 
                        'mean_squared_error', 'mean_absolute_error', 'root_mean_squared_error']:
            raise ValueError('Tuning must be one of: accuracy, f1, precision, recall, roc_auc, mean_squared_error, mean_absolute_error, root_mean_squared_error', 'root_mean_squared_error')
        self.sort_by = tuning
        neg_scoring = ['mean_absolute_error', 'mean_squared_error', 'mean_squared_error',
                        'log_loss']

        self.tuning = tuning if tuning not in neg_scoring else 'neg_' + tuning

        self.stacking = stacking
        self.name = name

        if self.tuning == 'neg_log_loss':
            #delete models that don't have predict_proba method
            for model in list(models_class.keys()):
                if not hasattr(models_class[model]['algo'], 'predict_proba'):
                    self.models.pop(model)

        dirs = ['params', 'trials', 'tables', 'data']
        for i in dirs:
            path = f'experiments/{self.name}/{i}'
            if not exists(path):
                os.makedirs(path)
                print(f'Directory {path} created!')

        if tuning != False and stacking is True:
            self.exp_type = 'stacking_tuning'
        elif tuning != False and stacking is False:
            self.exp_type = 'tuning'
        elif tuning is False and stacking is True:
            self.exp_type = 'stacking'
        else:
            self.exp_type = 'default'

        #if the results file exists, load it
        self.result_path = f'experiments/{self.name}/tables/{self.name}_{self.exp_type}_{tuning if tuning else ""}.csv'
        if exists(self.result_path):
            self.result = pd.read_csv(self.result_path, index_col = 0)
            print('Result loaded')

        for i in self.models.keys():
            # function that loads parameters from file
            file = f'experiments/{self.name}/params/{self.name}_{i}.pkl'
            if exists(file):
                with open(file, 'rb') as f:
                    self.models[i]['opt_params'] = pickle.load(f)
                    # print(f'Parameters for {i} loaded')

        self.dataset = [
                        f'experiments/{self.name}/data/X_train.csv',
                        f'experiments/{self.name}/data/y_train.csv',
                        f'experiments/{self.name}/data/X_test.csv',
                        f'experiments/{self.name}/data/y_test.csv'
                        ]
        #if the dataset exists, load it
        if all(exists(i) for i in self.dataset):
            self.X_train = pd.read_csv(self.dataset[0]).to_numpy() if array else pd.read_csv(self.dataset[0])
            self.y_train = pd.read_csv(self.dataset[1]).to_numpy().ravel() if array else pd.read_csv(self.dataset[1])
            self.X_test = pd.read_csv(self.dataset[2]).to_numpy() if array else pd.read_csv(self.dataset[2])
            self.y_test = pd.read_csv(self.dataset[3]).to_numpy().ravel() if array else pd.read_csv(self.dataset[3])
            print('Data loaded')


    def load_data(self, path, target, split=0.2, array=True):
        """
        Loads data from a csv file, splits it into train and test sets, 
        and copies it to the experiment/data folder.

        Args:
            path (str): Path to the csv file.
            target (str): Name of the target column.
            split (float, optional): Ratio between test and train set. Defaults to 0.2.
            array (bool, optional): Whether to convert the data to numpy arrays. Defaults to True.
        """
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

    @ignore_warnings(category=ConvergenceWarning)
    def start(self,  cv=5, raise_errors=False, n_eval=100, gpu = False, warnings=False):
        """Start the experiment. At the end of the experiment, the results are saved in the experiment/tables folder.

        Args:
            cv (int, optional): Number of folds for cross validation. Defaults to 5.
            raise_errors (bool, optional): Whether to raise errors. Defaults to False.
            n_eval (int, optional): Number of evaluations for hyperparameter tuning. Defaults to 100.
            gpu (bool, optional): Whether to use GPU. Only supported for XGBoost, CatBoost and LightGBM.
             Defaults to False.
            warnings (bool, optional): Whether to show warnings. Defaults to False.
        """
        #warnings off
        if warnings is False:
            import warnings
            warnings.filterwarnings("ignore")

        if gpu:
            self.models['XGBoost']['def_params']['tree_method'] = 'gpu_hist'
            self.models['XGBoost']['space']['tree_method'] = hp.choice('tree_method', ['gpu_hist'])

            # del self.models['XGBoost']['opt_params']['tree_method']
            self.models['LightGBM']['def_params']['device'] = 'gpu'
            self.models['LightGBM']['space']['device'] = hp.choice('device', ['gpu'])
            # del self.models['LightGBM']['opt_params']['device']
            # del self.models['CatBoost']['def_params']['task_type']
            # del self.models['CatBoost']['opt_params']['task_type']
        result = {
            'Model': self.models.keys(),
            }

    
        if self.result is not None:
            return
        scores_reg = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
        scores_multi = ['f1_weighted', 'neg_log_loss', 'accuracy']
        scores_bin = ['neg_log_loss', 'roc_auc', 'accuracy', 'precision', 'recall']

        if isinstance(self, RegressionExperiment):
            self.scoring = scores_reg
        else:
            self.n_classes = len(np.unique(self.y_train))
            self.scoring = scores_multi if self.n_classes > 2 else scores_bin
            if self.tuning == 'f1' and self.n_classes > 2:
                self.tuning = 'f1_weighted'
                self.sort_by = 'f1_weighted'

        for score_type in ['CV', 'Test', 'STD']:
            for score in self.scoring:
                result[f'{score_type} {score.replace("neg_", "")}'] = []

        if self.stacking:
            self.build_stack()
            self.models = self.stacking_models.copy()

        for (k, model_name) in enumerate(self.models.keys()):
            
            if isinstance(self, ClassificationExperiment) and 'neg_log_loss' not in self.scoring:
                self.scoring.append('neg_log_loss')

            current_time = datetime.now().strftime("%H:%M:%S")
            if self.stacking is True:
                model = self.models[model_name]

            if self.exp_type == 'tuning':
                
                model = self.models[model_name]

                file = f'experiments/{self.name}/params/{self.name}_{model_name}.pkl'
                if exists(file):
                    #print('🐫', model_name, 'already tuned 🐫')
                    with open(file, 'rb') as f:
                        params = pickle.load(f)
                else:
                    params = self.optimize_model(model_name=model_name, n_eval=n_eval)
                    clear_output(wait=True)
                    with open(file, 'wb+') as f:
                        pickle.dump(params, f)
                    print(file, 'saved')

                self.models[model_name]['opt_params'] = params.copy()
                # print('Optimized parameters', params)
                model = self.models[model_name]['algo'](**self.models[model_name]['opt_params'])

            else:
                model = self.models[model_name]['algo'](**self.models[model_name]['def_params'])
            clear_output(wait=True)
            print(f'{current_time}: 🐫 {k+1}/{len(self.models)+1} 🐫 Training {model_name} 🐫                    ', end='\r')
            
            
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            self.y_pred = y_pred

            if isinstance(self, ClassificationExperiment):
                try:
                    y_pred_proba = model.predict_proba(self.X_test)
                except:
                    y_pred_proba = None
                    self.scoring = [x for x in self.scoring if x != 'neg_log_loss']
                    result[f'CV log_loss'].append(np.nan)
                    result[f'STD log_loss'].append(np.nan)


            if cv:
                scores = cross_validate(model,
                                        self.X_train,
                                        self.y_train,
                                        scoring = self.scoring,
                                        cv=cv,
                                        n_jobs=self.n_j,
                                        error_score= 'raise'
                                        )

            for score in self.scoring:
                result[f'CV {score.replace("neg_","")}'].append(scores[f'test_{score}'].mean())
                result[f'STD {score.replace("neg_","")}'].append(scores[f'test_{score}'].std())


            self.result = result
            if self.X_test is not None:
                if isinstance(self, RegressionExperiment):
                    result['Test mean_squared_error'].append(mean_squared_error(self.y_test, y_pred))
                    result['Test root_mean_squared_error'].append(np.sqrt(mean_squared_error(self.y_test, y_pred)))
                    result['Test mean_absolute_error'].append(mean_absolute_error(self.y_test, y_pred))
                elif self.n_classes > 2:
                    result['Test f1_weighted'].append(metrics.f1_score(self.y_test, y_pred, average='weighted'))
                    result['Test accuracy'].append(metrics.accuracy_score(self.y_test, y_pred))
                    result['Test log_loss'].append(log_loss(self.y_test, y_pred_proba) if 'neg_log_loss' in self.scoring else np.nan)
                elif self.n_classes == 2:
                    result['Test roc_auc'].append(metrics.roc_auc_score(self.y_test, y_pred))
                    result['Test accuracy'].append(metrics.accuracy_score(self.y_test, y_pred))
                    result['Test precision'].append(metrics.precision_score(self.y_test, y_pred))
                    result['Test recall'].append(metrics.recall_score(self.y_test, y_pred))
                    result['Test log_loss'].append(log_loss(self.y_test, y_pred_proba) if 'neg_log_loss' in self.scoring else np.nan)

        result = pd.DataFrame(result)
        result_ind = result['Model']
        result.drop('Model', axis=1, inplace=True)
        result = result.apply(np.abs).apply(lambda x: round(x, 4))
        result.set_index(result_ind, inplace=True)
        asc = True if isinstance(self, RegressionExperiment) or self.sort_by == 'log_loss' else False
        sort_by = "CV " + self.sort_by if self.exp_type == "tuning" else result.columns[0]
        result = result.sort_values(by=sort_by, ascending=asc)
        
        self.result = result

        result.to_csv(self.result_path)
        clear_output(wait=True)

    def optimize_model(self, model_name, n_eval):
        """
        Function used to optimize the hyperparameters of a model using Hyperopt.

        Args:
            model_name (_type_): model to optimize.
            n_eval (_type_): number of evaluations.

        Returns:
            _type_: _description_
        """
        @ignore_warnings(category=ConvergenceWarning)
        def objective(space, model=self.models[model_name]['algo']):
            model = model(**space)
            now = datetime.now().strftime("%H:%M:%S")

            print(f'{now}: 🐫 Optimizing {model_name} 🐫                    ', end='\r')

            losses = model_selection.cross_val_score(
                model,
                self.X_train,
                self.y_train,
                scoring=self.tuning,
                n_jobs=self.n_j)

            clear_output(wait=True)


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

    def stack(self, n_estimators = 10, estimators = 'best'):
        """Start stacking process. This function trains a stacking and

        Args:
            n_estimators (int, optional): _description_. Defaults to 10.
            estimators (str, optional): One of "best", "all", "random".
            Best takes the n best models, all takes all models, random takes n random models.

        Returns:
            _type_: _description_
        """
        from .models import models_stacking_class, models_stacking_reg
        if isinstance(self, ClassificationExperiment):
            self.n_classes = len(np.unique(self.y_train))
            self.stacking_models = models_stacking_class
        else:
            self.stacking_models = models_stacking_reg

        self.stack_result_path = f'experiments/{self.name}/tables/{self.name}_stacking.csv'
        if exists(self.stack_result_path):
            self.stack_result = pd.read_csv(self.stack_result_path, index_col=0)
            self.stack_existing = pd.read_csv(self.stack_result_path, index_col=0)
        else:
            self.stack_result = pd.read_csv(self.result_path, nrows = 0 )
            self.stack_result = self.stack_result[[x for x in self.stack_result.columns if 'Test' in x]] 

        if estimators == 'best':
            self.stack_estimators = self.result.index[:n_estimators].to_list()
        elif estimators == 'all':
            self.stack_estimators = self.result.index.to_list()
        elif estimators == 'random':
            self.stack_estimators = np.random.choice(self.result.index.to_list(), n_estimators, replace = False)
        else:
            self.stack_estimators = estimators

        self.stack = []
        for model_name in self.stack_estimators:
            if model_name != 'CatBoost':
                try:
                    self.stack.append((model_name, self.models[model_name]['algo'](**self.models[model_name]['opt_params'])))
                except:
                    self.stack.append((model_name, self.models[model_name]['algo'](**self.models[model_name]['def_params'])))



        stacknames = []
        stackest = []
        self.predictions = []
        self.predictions_proba = []
        for est in self.models.values():
            clear_output()
            modello = est['algo'](**est['def_params']).fit(self.X_train, self.y_train)
            self.predictions.append(modello.predict(self.X_test))
            if hasattr(modello, 'predict_proba'):
                self.predictions_proba.append(modello.predict_proba(self.X_test))
            # self.predictions_proba.append(modello.predict_proba(self.X_test))
        self.stack_result = self.stack_result.to_dict()
        for i in self.stack_result:
            self.stack_result[i] = []

        for i in self.stacking_models.keys():
            self.stacking_models[i].estimators = self.stack
            if 'soft' in i:
                self.soft_models = [(i[0],i[1]) for i in self.stack if hasattr(i[1], 'predict_proba')]
                self.stacking_models[i].estimators = self.soft_models

            model = self.stacking_models[i].fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            #make a list of predictions for each model
            try:
                y_pred_proba = model.predict_proba(self.X_test)
            except:
                y_pred_proba = False
            stacknames.append(i)
            stackest.append(self.stack_estimators)

            if isinstance(self, RegressionExperiment):
                self.stack_result['Test mean_squared_error'].append(mean_squared_error(self.y_test, y_pred))
                self.stack_result['Test root_mean_squared_error'].append(np.sqrt(mean_squared_error(self.y_test, y_pred)))
                self.stack_result['Test mean_absolute_error'].append(mean_absolute_error(self.y_test, y_pred))
            elif self.n_classes > 2:
                self.stack_result['Test f1_weighted'].append(metrics.f1_score(self.y_test, y_pred, average='weighted'))
                self.stack_result['Test accuracy'].append(metrics.accuracy_score(self.y_test, y_pred))
                self.stack_result['Test log_loss'].append(log_loss(self.y_test, y_pred_proba) if y_pred_proba is not False else np.nan)
            elif self.n_classes == 2:
                self.stack_result['Test roc_auc'].append(metrics.roc_auc_score(self.y_test, y_pred))
                self.stack_result['Test accuracy'].append(metrics.accuracy_score(self.y_test, y_pred))
                self.stack_result['Test precision'].append(metrics.precision_score(self.y_test, y_pred))
                self.stack_result['Test recall'].append(metrics.recall_score(self.y_test, y_pred))
                self.stack_result['Test log_loss'].append(log_loss(self.y_test, y_pred_proba)  if y_pred_proba is not False else np.nan)
            
        self.stack_result['Model'] = stacknames
        self.stack_result['Estimators'] = stackest
        self.stack_result['N_estimators'] = [len(i) for i in stackest]
        self.stack_result = pd.DataFrame(self.stack_result)
        self.stack_result.set_index('Model', inplace=True)
        #concat
        if exists(self.stack_result_path):
            self.stack_result = pd.concat([self.stack_existing, self.stack_result])

        self.stack_result.to_csv(self.stack_result_path)
        return self.stack_result

class RegressionExperiment(Experiment):
    def __init__(self, name, tuning=False, stacking=False):
        self.stacking_models = models_stacking_reg
        self.models = models_reg
        super().__init__(name=name, tuning=tuning, stacking=stacking)


class ClassificationExperiment(Experiment):
    def __init__(self, name, tuning=False, stacking=False):
        self.stacking_models = models_stacking_class
        self.models = models_class
        super().__init__(name=name, tuning=tuning, stacking=stacking)
