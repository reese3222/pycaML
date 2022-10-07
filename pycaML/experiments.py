class Experiment():

    default = None
    tuned = None
    stacking = None
    names = []
    params = [] 
    spaces = []
    params = []
    trials = []
    parameters = []
    models_opt = []
    results_def = []
    stack_estimators = []
    
    #use all cores except one 
    n_j = -2

    def __init__(self, name, stacking = False, tuning = False, array = True):
        # name: name of the experiment
        # stacking: True or False

        self.tuning = tuning
        self.stacking = stacking
        import pickle
        from . import models_stacking_class, models_stacking_reg
        from . import models_class
        import os
        import shutil
        self.name = name
        import numpy as np
        if isinstance(self, BinaryClassificationExperiment):
            self.models = models_class
            self.stacking_models = models_stacking_class
            self.scoring =  ('f1', 'precision', 'recall')
            self.stacking_estimators_names_diverse = ['KNN', 'Logistic Regression', 'Decision Tree']

        elif isinstance(self, RegressionExperiment):
            self.stacking_models = models_stacking_reg
            from . import models_reg
            self.models = models_reg
            self.scoring = ['neg_mean_squared_error']
            self.stacking_estimators_names_diverse = ['KNN', 'Linear Regression', 'Decision Tree']

        else:
            self.stacking_models = models_stacking_class
            self.models = models_class  
            self.scoring = ['f1_weighted']
            self.stacking_estimators_names_diverse = ['KNN', 'Logistic Regression', 'Decision Tree']


        dirs = ['params', 'trials', 'tables','data']
        for i in dirs:
            path = f'experiments/{self.name}/{i}'
            if not os.path.exists(path):
                os.makedirs(path)
                print(f'Directory {path} created!')

        from os.path import exists
        import pandas as pd

        if tuning:
            file = f'experiments/{self.name}/tables/{self.name}_tuned.csv'
            if exists(file):
                self.tuned = pd.read_csv(file)
                print('Result loaded')
                if stacking is True:
                    best_boosting = self.tuned.loc[self.tuned['Model'].isin(['EBM', 'XGBoost', 'CatBoost', 'LightGBM', 'Gradient Boost'])].iloc[0,0]
                    best_bagging = self.tuned.loc[self.tuned['Model'].isin(['Random Forest', 'ExtraTrees'])].iloc[0,0]
                    self.stacking_estimators_names_diverse.append(best_bagging)
                    self.stacking_estimators_names_diverse.append(best_boosting)

        else:
            file = f'experiments/{self.name}/tables/{self.name}_def.csv'
            if exists(file):
                self.default = pd.read_csv(file)
                print('Result loaded')
                if stacking is True:
                    best_boosting = self.default.loc[self.default['Model'].isin(['EBM', 'XGBoost', 'CatBoost', 'LightGBM', 'Gradient Boost'])].iloc[0,0]
                    best_bagging = self.default.loc[self.default['Model'].isin(['Random Forest', 'ExtraTrees'])].iloc[0,0]
                    self.stacking_estimators_names_diverse.append(best_bagging)
                    self.stacking_estimators_names_diverse.append(best_boosting)

        
        for i in self.models.keys():
        #function that loads parameters from file
            file = f'experiments/{self.name}/params/{self.name}_{i}.pkl'
            if exists(file):
                with open(file, 'rb') as f:
                    self.models[i]['opt_params'] = pickle.load(f)
                    #print(f'Parameters for {i} loaded')
                    #print(self.models[i]['opt_params'] )
                    
        file = f'experiments/{self.name}/tables/{self.name}_stacking.csv'
        if exists(file):
            self.stacking = pd.read_csv(file)
            print('Result loaded')

        self.dataset = [
        f'experiments/{self.name}/data/X_train.csv',
        f'experiments/{self.name}/data/y_train.csv', 
        f'experiments/{self.name}/data/X_test.csv', 
        f'experiments/{self.name}/data/y_test.csv']

        if all(exists(i) for i in self.dataset):
            self.X_train = pd.read_csv(self.dataset[0]).to_numpy() if array else pd.read_csv(self.dataset[0])
            self.y_train = pd.read_csv(self.dataset[1]).to_numpy().ravel() if array else pd.read_csv(self.dataset[1])
            self.X_test = pd.read_csv(self.dataset[2]).to_numpy() if array else pd.read_csv(self.dataset[2])
            self.y_test = pd.read_csv(self.dataset[3]).to_numpy().ravel() if array else pd.read_csv(self.dataset[3])
            print('Data loaded')

    def load_data(self, path, target, split = 0.2, array = True):
        #import data from path

        import pandas as pd
        from sklearn.model_selection import train_test_split
        data = pd.read_csv(path)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.drop(target, axis = 1), data[target], test_size=split, random_state=42)
        self.X_train.to_csv(self.dataset[0], index = False)
        self.y_train.to_csv(self.dataset[1], index = False)
        self.X_test.to_csv(self.dataset[2], index = False)
        self.y_test.to_csv(self.dataset[3], index = False)
        print('Data loaded')

    def warn(self, warn = 'on'):
        import sys
        if not sys.warnoptions and warn == 'off':
            import warnings
            warnings.simplefilter("ignore")

    

    #trains the models and saves the results  
    def start(self,  cv = 5, raise_errors = False, n_eval = 100):
        import pandas as pd
        from sklearn import metrics
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_validate
        from tqdm import tqdm
        import numpy as np
        from statistics import mean
        import pickle
        import os

        

        tables_path  = f'experiments/{self.name}/tables/{self.name}'

        if self.tuning is False and self.stacking is False:
            if isinstance(self, RegressionExperiment):
                self.models['Linear Regression']['algo'] = LinearRegression
                self.models['Linear Regression']['def_params'] = {}
            if os.path.exists(f'{tables_path}_def.csv'):
                return None
        elif self.tuning is False and self.stacking is True:
            if isinstance(self, RegressionExperiment):
                self.models['Linear Regression']['algo'] = LinearRegression
                self.models['Linear Regression']['def_params'] = {}

            if os.path.exists(f'{tables_path}_def_stacking.csv'):
                return None
            #load default table
            elif os.path.exists(f'{tables_path}_def.csv'):
                self.default = pd.read_csv(f'{tables_path}_def.csv')
            else:
                self.stacking = False


        elif self.tuning is True and self.stacking is False:
            if os.path.exists(f'{tables_path}_tuned.csv'):
                return None
        elif self.tuning is True and self.stacking is True:
            if os.path.exists(f'{tables_path}_tuned_stacking.csv'):
                return None
            #load default table
            elif os.path.exists(f'{tables_path}_tuned.csv'):
                self.tuned = pd.read_csv(f'{tables_path}_tuned.csv')
            else:
                self.stacking = False



        if self.stacking is True:
            self.build_stack()
            self.models = self.stacking_models.copy()


        if isinstance(self, RegressionExperiment):
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

        for (k, model_name) in enumerate(self.models.keys()):
            from datetime import datetime
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            if self.stacking == True:
                model = self.models[model_name]
                model_algo = model['algo']

            if self.tuning is True  and self.stacking == False:
                model = self.models[model_name]
                print(current_time, ': 📘 Optimizing', model_name, '📘')

                file = f'experiments/{self.name}/params/{self.name}_{model_name}.pkl'
                if os.path.exists(file):
                    print('📘', model_name, 'already tuned 📘')
                    with open(file, 'rb') as f:
                        params = pickle.load(f)
                else:
                    if raise_errors:
                        params = self.optimize_model(model_name = model_name, n_eval = n_eval)
                        with open(file, 'wb+') as f:
                            pickle.dump(params, f)
                        print(file, 'saved')
                    else:
                        params = self.optimize_model(model_name = model_name,n_eval = n_eval)
                        with open(file, 'wb+') as f:
                            pickle.dump(params, f)
                        print(file, 'saved')

                self.models[model_name]['opt_params'] = params.copy()
                #print('Optimized parameters', params)
                
                model = self.models[model_name]['algo'](**self.models[model_name]['opt_params'])
            
            else:
                model = self.models[model_name]['algo'](**self.models[model_name]['def_params'])

            n_j = 1 if model_name in ['XGB', 'LGBM'] else self.n_j

            print(f'{current_time}: Training {model_name} {k+1}/{len(self.models)+1}          ', end = '\r')
            scores = cross_validate(model, self.X_train, self.y_train, scoring =  self.scoring, cv = cv, n_jobs=n_j, error_score="raise")

            model.fit(self.X_train, self.y_train )
            y_pred = model.predict(self.X_test)
            self.y_pred = y_pred

            if isinstance(self, RegressionExperiment):
                mean_mse.append(np.sqrt(-mean(scores['test_neg_mean_squared_error'])))
                mse_test.append(np.sqrt(metrics.mean_squared_error(self.y_test, y_pred)))
                duration_time.append(mean(scores['fit_time']))
                std_mse.append(np.std(np.sqrt(-scores['test_neg_mean_squared_error'])))
            elif self.task == 'multiclass':
                mean_F1.append(mean(scores['test_f1_weighted']))
                std_F1.append(np.std(scores['test_f1_weighted']))
                F1_test.append(metrics.f1_score(self.y_test, y_pred, average = 'weighted'))
                duration_time.append(mean(scores['fit_time']))
            else:
                mean_F1.append(mean(scores['test_f1']))
                std_F1.append(np.std(scores['test_f1']))
                F1_test.append(metrics.f1_score(self.y_test, y_pred))
                duration_time.append(mean(scores['fit_time']))
                precision.append(metrics.precision_score(self.y_test, y_pred))
                recall.append(metrics.recall_score(self.y_test, y_pred))

        if isinstance(self, RegressionExperiment):
            result  = {
                'Model': self.models.keys(), 
                'RMSE_test': mse_test, 
                'RMSE_cv': mean_mse, 
                'rmse_Std' : std_mse,
                'Time' : duration_time,
            }

            result = pd.DataFrame(result).sort_values(by = 'RMSE_test', ascending= True)

        else:
            result  = {
            'Model': self.models.keys(), 
            'F1_test': F1_test, 
            'F1_cv': mean_F1, 
            'F1_Std' : std_F1,
            'Time' : duration_time,
            }
            if isinstance(self, BinaryClassificationExperiment):
                result['Precision_test'] =  precision
                result['Recall_test'] = recall

            result = pd.DataFrame(result).sort_values(by = 'F1_test', ascending= False)
        
        self.result = result
        if self.tuning is True  and self.stacking == False:
            self.tuned = result
            result.to_csv(f'{tables_path}_tuned.csv', index = False)

        elif self.tuning is True  and self.stacking == True:
            self.stacking_tuned = result
            result.to_csv(f'{tables_path}_tuned_stacking.csv', index = False)

        elif self.tuning is False and self.stacking == False:
            self.default = result
            result.to_csv(f'{tables_path}_def.csv',index = False)

        elif self.tuning is False and self.stacking == True:
            self.stacking_default = result
            result.to_csv(f'{tables_path}_def_stacking.csv',index = False)    

    #setup creates folders for the experiment
    def setup(self):
        import os
        import shutil

        dirs = ['params', 'trials', 'tables','data']
        for i in dirs:
            path = f'experiments/{self.name}/{i}'
            if not os.path.exists(path):
                os.makedirs(path)
                print(f'Directory {path} created!')
            else:
                print(f'Directory {path} already exists!')
        target = f'experiments/{self.name}/{self.name}.csv'
 

    def optimize_model(self, model_name, n_eval):
        import pickle
        from statistics import mean
        import numpy as np
        from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
        from sklearn import model_selection
        def objective(space, model = self.models[model_name]['algo']):
            model = model(**space)
            losses = model_selection.cross_val_score(model,self.X_train, self.y_train, scoring=self.scoring[0], n_jobs = self.n_j) 
            return {'status': STATUS_OK,
            'loss': -mean(losses),
            'loss_variance': np.var(losses, ddof=1)}

        trial = Trials()
        best = fmin(objective, space = self.models[model_name]['space'], algo=tpe.suggest, max_evals=n_eval, trials = trial, show_progressbar = True)
        file =  'experiments/' + self.name + '/trials/trial_' + self.name + '_'+  model_name + '.pkl'
        with open(file, 'wb+') as f:
            pickle.dump(trial, f)
        return space_eval(self.models[model_name]['space'], best)
    
    #optimize every model using TPE (Bayesian Optimization)
    def optimize(self, raise_errors = True, n_eval = 100):
        import pickle
        from os.path import exists
        for model_name, model in self.models.items():
            #check if the model has been already tuned
            file =  'experiments/' + self.name + '/params/params_' + self.name + '_'+  model_name + '.pkl'
            if exists(file):
                model_params = pickle.load(open(file, 'rb'))
                self.models[model_name]['opt_params'] = model(**model_params)
                print('Found parameters for', model_name)
            else:
                print('Optimizing', model_name)
                if raise_errors:
                    model_params = self.optimize_model(
                        model_name = model_name, 
                        name=self.name, 
                        model = self.models[model_name]['algo'], 
                        space =  self.models[model_name]['space'], 
                        max_eval = n_eval, 
                        scoring_opt = self.scoring_opt[0])
                else:
                    try:
                        model_params = self.optimize_model(
                            model_name = model_name, 
                            name=self.name, 
                            model = self.models[model_name]['algo'], 
                            space =  self.models[model_name]['space'], 
                            max_eval = n_eval, 
                            scoring_opt = self.scoring_opt[0])
                    except:
                        print('Optimization failed for', model_name)
                with open(file, 'wb+') as f:
                    pickle.dump(model_params, f)
                print(file, 'saved')
                self.models[model_name]['opt_params'] = model(**model_params)

    #method used for passing the parameters to the model. Stacking diverse models takes the best model from boosting and the best model from bagging
    def build_stack(self):
        if self.tuning is False:
            estimators_stacking_all = [(i, j['algo'](**self.models[i]['def_params'])) for i,j in self.models.items() ]
        else:
            estimators_stacking_all = [(i, j['algo'](**self.models[i]['opt_params'])) for i,j in self.models.items() ]

        self.stacking_models['Stacking (all)']['def_params']['estimators'] = estimators_stacking_all
        self.stacking_models['Voting (all)']['def_params']['estimators'] = estimators_stacking_all
        self.stacking_models_diverse = {m: self.models[m] for m in self.stacking_estimators_names_diverse}

        if self.tuning is False:
            estimators_stacking_diverse = [(i, j['algo'](**self.models[i]['def_params'])) for i,j in self.stacking_models_diverse.items() ]
        else:
            estimators_stacking_diverse = [(i, j['algo'](**self.models[i]['opt_params'])) for i,j in self.stacking_models_diverse.items() ]

        self.stacking_models['Stacking (diverse)']['def_params']['estimators'] = estimators_stacking_diverse
        self.stacking_models['Voting (diverse)']['def_params']['estimators'] = estimators_stacking_diverse

class RegressionExperiment(Experiment):
    def __init__(self, name, tuning = False, stacking = False):
        super().__init__(name = name, tuning = tuning, stacking = stacking)

class BinaryClassificationExperiment(Experiment):
    def __init__(self, name, tuning = False, stacking = False):
        super().__init__(name = name, tuning = tuning, stacking = stacking)

class MulticlassExperiment(Experiment):
    def __init__(self, name, tuning = False, stacking = False):
        super().__init__(name = name, tuning = tuning, stacking = stacking)