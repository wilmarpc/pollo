"""Module that lists specific model classes to be used in the `ModelZoo`.
"""

import abc
import numpy as np
import pandas as pd
import numpy as np
import joblib
from celonis_ml.modeling import metrics
from celonis_ml.helpers import grid_search_cv, random_search_cv, custom_asymmetric_valid
from hyperopt.pyll.base import scope
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, roc_curve, mean_squared_error, roc_auc_score, f1_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import lightgbm as lgbm
import logging

AVAILABLE_MODELS = [
    "lgbm_classifier",
    "lgbm_regressor",
    "lr_classifier",
    "rf_classifier",
    "xgb_classifier"
]


PY_MODELS_MASTER = pd.DataFrame({
    'model_name': ['lgbm_classifier', 'lgbm_regressor', 'rf_classifier', 'lr_classifier', 'xgb_classifier'],
    'model_number': [1, 2, 3, 4, 5],
    'long_name': ['LightGBM Classifier', 'LightGBM Regressor', 'Random Forest Classifier',
    'Logistic Regression Classifier', 'XGB Classifier']
})



class Model():
    """
    Abstract Model class.
    """

    __metaclass__ = abc.ABCMeta

    model_instance = None
    type = None
    name = None
    args = None
    info = None
    data_dir = None

    @abc.abstractmethod
    def initialize(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def set_training_data(self, X_train, y_train):
        raise NotImplementedError

    @abc.abstractmethod
    def set_validation_data(self, X_valid, y_valid, reference):
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, train, valid=None, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, model_path, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, model_path, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_feature_importance(self, X, **kwargs):
        raise NotImplementedError

    @classmethod
    def _add_description_factory(cls, descr):
        """
        Factory method which populates the un-set class variables.

        Returns
        -------
          new model class
        """
        for field in ['type', 'name', 'info', 'data_dir']:
            setattr(cls, field, getattr(descr, field))
        return cls


class LogisticRegressionClassifier(Model):
    """
    Specific class for a Logistic Regression classifier
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.name = "lr_classifier"
        self.metrics = metrics.classification_scoring
        self.type = "classifier"
        self.classes = None
        self.X = None
        self.y = None

    def initialize(self, **kwargs):
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(
            random_state=0,
            penalty='l1',
            C=1e-2,
            n_jobs=-1)
        self.model_instance = classifier
        if self.classes is not None:
            self.model_instance.classes_ = self.classes
            if len(self.classes) > 2:
                self.model_instance.multi_class = "multinomial"
                self.model_instance.solver = "saga"

    def set_training_data(self, X_train, y_train):
        train = {'X': X_train, 'y': y_train}
        self.X = X_train
        self.y = y_train
        if y_train.dtype == 'object':
            cats = y_train.astype('category')
            y_train = cats.cat.codes
            self.classes = cats.dtype.categories.values
        return train, self.classes

    def set_validation_data(self, X_valid, y_valid, reference):
        valid = {'X': X_valid, 'y': y_valid}
        return valid

    def fit(self, train, valid=None):
        X, y = train.get('X'), train.get('y')
        self.model_instance.fit(X, y)

    def predict(self, X, **kwargs):
        y_predicted = self.model_instance.predict_proba(X)
        if hasattr(self.model_instance, 'classes_') and set(self.model_instance.classes_) != {0, 1}:
            # Multi-class classification
            return y_predicted
        else:
            # Binary classification
            return y_predicted[:, 1]

    def save_model(self, model_path, **kwargs):
        joblib.dump(self.model_instance, model_path)

    @classmethod
    def load_model(self, model_path, **kwargs):
        return joblib.load(model_path)

    def get_feature_importance(self, **kwargs):
        feature_importance = pd.DataFrame({"feature_name": self.X.columns.values,
                                           "metric": np.full((1, len(self.X.columns)), "coefficient")[0],
                                           "importance": self.model_instance.coef_[0],
                                           "model_name": np.full((1, len(self.X.columns)), self.name)[0]})

        return feature_importance


class LGBMClassifier(Model):
    """
    Specific class for a LGBM classifier
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.X = None
        self.y = None
        self.train = None
        self.num_class = None
        self.name = "lgbm_classifier"
        self.type = "classifier"
        self.metrics = metrics.classification_scoring
        self.params = {'num_leaves': 64, 'num_trees': 750, 'objective': 'binary', 'metric': 'auc',
                       'learning_rate': 0.01, 'max_bin': 256, "num_rounds": 100, "min_gain_to_split":0.1}
        self.cv = 5
        self.max_evals = 5
        self.best_loss = None
        self.best_hyperparams = None
        self._logger = logging.getLogger(__name__)

        self.feature_importance = None
        self.feature_importance_metrics = ['gain', 'split']

        self.hyper_param_search_space = {
            'multiclass': {
                'num_leaves': hp.choice('num_leaves', np.arange(2, 32, dtype=int)),
                'max_depth': hp.choice('max_depth', np.arange(3, 8, dtype=int)),
                'learning_rate': hp.choice('learning_rate', [0.001, 0.005, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0]),
                'num_rounds': hp.choice('num_rounds', np.arange(500, 1500, 100, dtype=int)),
                'objective': 'multiclass',
            },
            'binary': {
                'num_leaves': hp.choice('num_leaves', np.arange(2, 32, dtype=int)),
                'max_depth': hp.choice('max_depth', np.arange(3, 8, dtype=int)),
                'learning_rate': hp.choice('learning_rate', [0.001, 0.005, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0]),
                'num_rounds': hp.choice('num_rounds', np.arange(500, 1500, 100, dtype=int)),
                'scale_pos_weight': hp.choice('scale_pos_weight', np.arange(1, 12, 2, dtype=int)),
                'objective': 'binary',
            }
        }
        

    def initialize(self, **kwargs):
        classifier = lgbm.LGBMClassifier(
            n_jobs=-1,
            random_state=33,
            **self.params
        )
        self.model_instance = classifier

    def set_training_data(self, X_train, y_train):
        self.num_class = len(np.unique(y_train))
        classes = None

        if self.num_class > 2:
            self.params["objective"] = 'multiclass'
            self.params["num_class"] = self.num_class
            self.params["metric"] = ['multi_logloss']
        if y_train.dtype == 'object':
            cats = y_train.astype('category')
            y_train = cats.cat.codes
            classes = cats.dtype.categories.values
        
        import lightgbm as lgbm
        train = lgbm.Dataset(X_train, y_train, free_raw_data=False)

        self.X = X_train
        self.y = y_train
        self.train = train
        return train, classes

    def set_validation_data(self, X_valid, y_valid, reference):
        import lightgbm as lgbm

        if y_valid.dtype == 'object':
            cats = y_valid.astype('category')
            y_valid = cats.cat.codes

        valid = lgbm.Dataset(X_valid, y_valid, reference=reference,
                             free_raw_data=False)
        return valid

    def fit(self, train, tuned_params=None):
        params = self.params

        if tuned_params:
            params.update(tuned_params)

        self.model_instance = lgbm.train(
            params=params,
            train_set=train,
            init_model=self.model_instance,
            verbose_eval=False)
            # feval=custom_asymmetric_valid)

    def predict(self, X, **kwargs):
        y_predicted = self.model_instance.predict(X)
        return y_predicted

    def save_model(self, model_path, **kwargs):
        joblib.dump(self.model_instance, model_path)

    @classmethod
    def load_model(self, model_path, **kwargs):
        return joblib.load(model_path)

    def get_feature_importance(self, **kwargs):
        feature_importance = None

        for metric in self.feature_importance_metrics:
            partial_feature_importance = pd.DataFrame({"feature_name": self.X.columns.values,
                                                       "metric": np.full((1, len(self.X.columns)), metric)[0],
                                                       "importance": self.model_instance.feature_importance(importance_type=metric, iteration=None),
                                                       "model_name": np.full((1, len(self.X.columns)), self.name)[0]})

            feature_importance = partial_feature_importance if feature_importance is None else pd.concat([feature_importance, partial_feature_importance])
        
        return feature_importance


    def objective_function(self, args):
        """
        This function used to minimize the calculatud loss in each iteration of the hyper parameter optimization.
        For this a cross validation is done using a scorer to calculate the loss.
        If the loss of the current evaluation is better than the best loss, the best parameters are updated.

        Parameters
        ----------
        args : dict
            Current evaluated parameter combination

        Returns
        -------
        int
            loss of the current evaluated hyper parameter combination
        """

        self.model_instance.set_params(**args)
        if self.num_class > 2:
            cv_results = lgbm.cv(params=self.model_instance.get_params(), train_set=self.train, stratified=True, nfold=self.cv, metrics='multi_logloss', early_stopping_rounds=10)
            best_score = min(cv_results['multi_logloss-mean'])
        else:
            cv_results = lgbm.cv(params=self.model_instance.get_params(), train_set=self.train, stratified=True, nfold=self.cv, metrics='auc', early_stopping_rounds=10)
            best_score = 1 - max(cv_results['auc-mean'])
        loss = best_score

        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.best_hyperparams = self.model_instance.get_params()
        self._logger.info(f"Best Hyperparameters: {self.best_hyperparams}, Best Loss:{self.best_loss}")
        return loss

    def optimize_model(self, **kwargs):
        """
        This function optimizes the parameters using the fmin() function of hyperopt using the specified hyperparameter search space
        and returns the best parameters.

        Parameters for fmin()
        ----------
        objective_function : function
            Function used to minimize the calculatud loss in each iteration.
        hyper_param_search_space : dict
            Search space of alle parameters which shall be optimized.
        algo : function
            Search algorithm (Default: tpe.suggest - Tree-structured Parzen estimator )
            optional: random.suggest - Random search
        max_evals :
            Number of iteration rounds.
        trials: hyperopt.base.Trials
            Database in which to store all the point evaluations of the search
        Returns
        -------
        dict
            result of the best parameters after optimizing all parameters
        """
        if self.num_class > 2:
            search_space = self.hyper_param_search_space['multiclass']

        else:
            search_space = self.hyper_param_search_space['binary']

        if kwargs.get('cv_rounds'):
            self.cv = kwargs.get('cv_rounds')
        
        if kwargs.get('max_evals'):
            self.max_evals = kwargs.get('max_evals')


        trials = Trials()
        best = fmin(self.objective_function, search_space,
                    algo=tpe.suggest, max_evals=self.max_evals, trials=trials, return_argmin=False)
        return best


class RFClassifier(Model):
    """
    Specific class for a Random Forest classifier
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.X = None
        self.y = None
        self.name = "rf_classifier"
        self.metrics = metrics.classification_scoring
        self.type = "classifier"
        self.classes = None
        self._logger = logging.getLogger(__name__)
        self.params = {'max_leaf_nodes': None, 'n_estimators': 1400, 'max_depth': 10, 'min_samples_split': 3, 
                       'criterion': 'gini', "class_weight": 'balanced', 'min_samples_leaf': 1, 'max_features': 'auto',
                       'oob_score': True}
        self.cv = 5
        self.max_evals = 5
        self.best_loss = None
        self.best_hyperparams = None
        self.hyper_param_search_space = {
            'max_depth': hp.choice('max_depth', np.arange(3, 10, 2, dtype=int)),
            'n_estimators': hp.choice('n_estimators', np.arange(500, 1500, 100, dtype=int)),
            'class_weight': hp.choice('class_weight', ['balanced']),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', 0.2, 0.5, 0.8]),
            'min_samples_split': hp.choice('min_samples_split', [2, 5, 20, 50]),
            'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4]),

        }

    def initialize(self, **kwargs):
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(**self.params, n_jobs=-1)
        self.model_instance = classifier
        if self.classes is not None:
            self.model_instance.classes_ = self.classes

    def set_training_data(self, X_train, y_train):
        train = {'X': X_train, 'y': y_train}
        self.num_class = len(np.unique(y_train))
        self.X = X_train
        self.y = y_train
        if y_train.dtype == 'object':
            cats = y_train.astype('category')
            y_train = cats.cat.codes
            self.classes = cats.dtype.categories.values

        return train, self.classes

        


    def set_validation_data(self, X_valid, y_valid, reference):
        valid = {'X': X_valid, 'y': y_valid}
        return valid


    def fit(self, train, valid=None, train_mode='NoGrid', tuned_params=None):
        from sklearn.ensemble import RandomForestClassifier
        params = self.params

        if tuned_params:
            params = tuned_params
        
        X, y = train.get('X'), train.get('y')
        
        classifier = RandomForestClassifier(**params, n_jobs=-1)
        self.model_instance = classifier
        self.model_instance.fit(X, y)

    def predict(self, X, **kwargs):
        y_predicted = self.model_instance.predict_proba(X)
        if hasattr(self.model_instance, 'classes_') and set(self.model_instance.classes_) != {0, 1}:
            # Multi-class classification
            return y_predicted
        else:
            # Binary classification
            return y_predicted[:, 1]

    def save_model(self, model_path, **kwargs):
        joblib.dump(self.model_instance, model_path)

    @classmethod
    def load_model(self, model_path, **kwargs):
        return joblib.load(model_path)

    def objective_function(self, args):
        """
        This function used to minimize the calculatud loss in each iteration of the hyper parameter optimization.
        For this a cross validation is done using a scorer to calculate the loss.
        If the loss of the current evaluation is better than the best loss, the best parameters are updated.

        Parameters
        ----------
        args : dict
            Current evaluated parameter combination

        Returns
        -------
        int
            loss of the current evaluated hyper parameter combination
        """

        self.model_instance.set_params(**args)
        if self.num_class > 2:
            scorer = make_scorer(
                f1_score, average='macro', greater_is_better=True)
        else:
            scorer = make_scorer(
                roc_auc_score, average='weighted', greater_is_better=True)
        strat_fold = StratifiedKFold(n_splits=self.cv, shuffle=True)
        cv_results = cross_validate(
            self.model_instance, self.X, self.y, scoring=scorer, cv=strat_fold)
        best_score = (cv_results['test_score']).mean()
        loss = 1 - best_score

        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.best_hyperparams = args

        self._logger.info(f"Best Hyperparameters: {self.best_hyperparams}, Best Loss:{self.best_loss}")
        return loss

    def optimize_model(self):
        """
        This function optimizes the parameters using the fmin() function of hyperopt using the specified hyperparameter search space
        and returns the best parameters.

        Parameters for fmin()
        ----------
        objective_function : function
            Function used to minimize the calculatud loss in each iteration.
        hyper_param_search_space : dict
            Search space of alle parameters which shall be optimized.
        algo : function
            Search algorithm (Default: tpe.suggest - Tree-structured Parzen estimator )
            optional: random.suggest - Random search
        max_evals :
            Number of iteration rounds.
        trials: hyperopt.base.Trials
            Database in which to store all the point evaluations of the search
        Returns
        -------
        dict
            result of the best parameters after optimizing all parameters
        """
        

        trials = Trials()
        best = fmin(self.objective_function, self.hyper_param_search_space,
                    algo=tpe.suggest, max_evals=self.max_evals, trials=trials, return_argmin=False)
        return best



    def optimize_model_new(self):


        from sklearn.model_selection import RandomizedSearchCV
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
        

        rf = self.model_instance
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3,
                                       verbose=2, random_state=42, n_jobs = -1)
        # Fit the random search model
        rf_random.fit(self.X, self.y)

        best = rf_random.best_params_

        self._logger.info(f"Best Hyperparameters: {best}")

        return best

    def get_feature_importance(self, **kwargs):
            feature_importance = pd.DataFrame({"feature_name": self.X.columns.values,
                                                        "metric": np.full((1, len(self.X.columns)), "impurity")[0],
                                                        "importance": self.model_instance.feature_importances_,
                                                        "model_name": np.full((1, len(self.X.columns)), self.name)[0]})
          
            return feature_importance


class XGBClassifier(Model):
    """
    Specific class for a Extreme Gradient Boosting classifier
    TODO: Find value error
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.name = "xgb_classifier"
        self.metrics = metrics.classification_scoring
        self.type = "classifier"

    def initialize(self, **kwargs):
        import xgboost as xgb
        classifier = xgb.XGBClassifier()
        self.model_instance = classifier

    def set_training_data(self, X_train, y_train):
        train = {'X': X_train, 'y': y_train}
        return train

    def set_validation_data(self, X_valid, y_valid, reference):
        valid = {'X': X_valid, 'y': y_valid}
        return valid

    def fit(self, train, valid=None, train_mode='NoGrid'):
        X, y = train.get('X'), train.get('y')
        self.model_instance.fit(X, y)

    def predict(self, X, **kwargs):
        y_predicted = self.model_instance.predict_proba(X)
        y_predicted = y_predicted[:, 1]
        return(y_predicted)

    def save_model(self, model_path, **kwargs):
        joblib.dump(self.model_instance, model_path)

    @classmethod
    def load_model(self, model_path, **kwargs):
        return joblib.load(model_path)


class KNNClassifier(Model):
    """
    Specific class for a K-Nearest Neighbor classifier
    """
    # TODO: Don't split dataset into training/valid, whole dataset is used to find most similar instances
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.name = "knn__classifier"
        self.metrics = metrics.classification_scoring
        self.type = "classifier"

    def initialize(self, **kwargs):
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=10)
        self.model_instance = classifier

    def set_training_data(self, X_train, y_train):
        train = {'X': X_train, 'y': y_train}
        return train

    def set_validation_data(self, X_valid, y_valid, reference):
        valid = {'X': X_valid, 'y': y_valid}
        return valid

    def fit(self, train, valid=None, train_mode='NoGrid'):
        X, y = train.get('X'), train.get('y')
        self.model_instance.fit(X, y)

    def predict(self, X, **kwargs):
        # TODO: Implement mandatory dim-reduction for KNN due to curse of dimensionality for euclidean distance
        y_predicted = self.model_instance.predict_proba(X)
        y_predicted = y_predicted[:, 1]
        return(y_predicted)

    def save_model(self, model_path, **kwargs):
        joblib.dump(self.model_instance, model_path)

    @classmethod
    def load_model(self, model_path, **kwargs):
        return joblib.load(model_path)


class MLPClassifier(Model):

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.name = 'mlp__classifier'
        self.type = "classifier"

    def initialize(self, **kwargs):
        # TODO: use upcoming class weight support https://github.com/scikit-learn/scikit-learn/pull/11723
        from sklearn import neural_network
        self.model_instance = neural_network.MLPClassifier(
            hidden_layer_sizes=[512, 256, 128, 64])

    def set_training_data(self, X_train, y_train):
        train = {'X': X_train, 'y': y_train}
        return train

    def set_validation_data(self, X_valid, y_valid, reference):
        valid = {'X': X_valid, 'y': y_valid}
        return valid

    def fit(self, train, valid=None):
        X, y = train.get('X'), train.get('y')
        self.model_instance.fit(X, y)

    def predict(self, X, **kwargs):
        return self.model_instance.predict(X)

    def save_model(self, model_path, **kwargs):
        joblib.dump(self.model_instance, model_path)

    @classmethod
    def load_model(self, model_path, **kwargs):
        return joblib.load(model_path)


class LGBMRegressor(Model):
    """
    Specific class for a LGBM regressor
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.X = None
        self.y = None
        self.train = None

        self.name = "lgbm_regressor"
        self.type = "regressor"
        self.metrics = metrics.regression_scoring
        self.params = {'num_leaves': 100, 'num_trees': 100, 'objective': 'regression',
                       'learning_rate': 0.1, 'max_bin': 256, "num_rounds": 100}
        self.cv = 5
        self.max_evals = 5
        self.best_loss = None
        self.best_hyperparams = None
        self._logger = logging.getLogger(__name__)
        self.feature_importance = None
        self.feature_importance_metrics = ['gain', 'split']


        self.hyper_param_search_space = {
            'num_leaves': hp.choice('num_leaves', np.arange(30, 100, dtype=int)),
            'max_depth': hp.choice('max_depth', np.arange(0, 6, dtype=int)),
            'learning_rate': hp.choice('learning_rate', [0.001, 0.005, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0]),
            'num_rounds': hp.choice('num_rounds', np.arange(500, 1500, 100, dtype=int)),
            'objective': 'regression',
        }

    def initialize(self, **kwargs):
        regressor = lgbm.LGBMRegressor(
            n_jobs=-1,
            random_state=33,
            **self.params
        )
        self.model_instance = regressor

        return regressor

    def set_training_data(self, X_train, y_train):
        train = lgbm.Dataset(X_train, y_train, free_raw_data=False)
        self.X = X_train
        self.y = y_train
        self.train = train
        return train

    def set_validation_data(self, X_valid, y_valid, reference):
        valid = lgbm.Dataset(X_valid, y_valid, reference=reference,
                             free_raw_data=False)
        return valid

    def fit(self, train, valid=None, tuned_params=None):
        params = self.params

        if tuned_params:
            params = tuned_params

        self.model_instance = lgbm.train(
            params=params,
            train_set=train,
            init_model=self.model_instance,
            verbose_eval=False)
            #feval=custom_asymmetric_valid)

    def predict(self, X, **kwargs):
        y_predicted = self.model_instance.predict(X)
        return y_predicted

    def save_model(self, model_path, **kwargs):
        joblib.dump(self.model_instance, model_path)

    @classmethod
    def load_model(self, model_path, **kwargs):
        return joblib.load(model_path)

    def objective_function(self, args):
        """
        This function used to minimize the calculatud loss in each iteration of the hyper parameter optimization.
        For this a cross validation is done using a scorer to calculate the loss. 
        If the loss of the current evaluation is better than the best loss, the best parameters are updated.

        Parameters
        ----------
        args : dict
            Current evaluated parameter combination
      
        Returns
        -------from sklearn.utils import class_weight
In order to calculate the class weight do the following

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
        int 
            loss of the current evaluated hyper parameter combination
        """

        self.model_instance.set_params(**args)
        cv_results = lgbm.cv(params=self.model_instance.get_params(), train_set=self.train, stratified=True, nfold=self.cv, metrics='l2', early_stopping_rounds=10)
        best_score = min(cv_results['l2-mean'])
        loss = np.sqrt(best_score)

        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.best_hyperparams = self.model_instance.get_params()
        self._logger.info(f"Best Hyperparameters: {self.best_hyperparams}, Best Loss:{self.best_loss}")
        return loss

    def optimize_model(self, **kwargs):
        """
        This function optimizes the parameters using the fmin() function of hyperopt using the specified hyperparameter search space
        and returns the best parameters.

        Parameters for fmin()
        ----------
        objective_function : function
            Function used to minimize the calculatud loss in each iteration.
        hyper_param_search_space : dict
            Search space of alle parameters which shall be optimized.
        algo : function
            Search algorithm (Default: tpe.suggest - Tree-structured Parzen estimator )
            optional: random.suggest - Random search
        max_evals :
            Number of iteration rounds.
        trials: hyperopt.base.Trials
            Database in which to store all the point evaluations of the search
        Returns
        -------from sklearn.utils import class_weight

        dict
            result of the best parameters after optimizing all parameters
        """
        if kwargs.get('cv_rounds'):
            self.cv = kwargs.get('cv_rounds')
        
        if kwargs.get('max_evals'):
            self.max_evals = kwargs.get('max_evals')


        trials = Trials()
        best = fmin(self.objective_function, self.hyper_param_search_space,
                    algo=tpe.suggest, max_evals=self.max_evals, trials=trials, return_argmin=False)

        return best

    def get_feature_importance(self, **kwargs):
        feature_importance = None

        for metric in self.feature_importance_metrics:
            partial_feature_importance = pd.DataFrame({"feature_name": self.X.columns.values,
                                                       "metric": np.full((1, len(self.X.columns)), metric)[0],
                                                       "importance": self.model_instance.feature_importance(importance_type=metric, iteration=None),
                                                       "model_name": np.full((1, len(self.X.columns)), self.name)[0]})

            feature_importance = partial_feature_importance if feature_importance is None else pd.concat([feature_importance, partial_feature_importance])
        
        return feature_importance


       

    def optimize_model_random_search(self):

        from sklearn.model_selection import RandomizedSearchCV
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
        

        rf = self.model_instance
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        # Fit the random search model
        rf_random.fit(self.X, self.y)

        best = rf_random.best_params_

        return best
