import datetime as datetime
import getpass
import glob
import logging
import os
import time
from pathlib import Path
from time import gmtime, strftime

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import model_selection

from celonis_ml.data_preprocessing import DataLoader
from celonis_ml.data_preprocessing.data_utils import dump_data_file, load_data_file, verify_pred_set
from celonis_ml.helpers import create_path, performance_at_threshold, performance_classification, performance_regression
from celonis_ml.modeling import AVAILABLE_MODELS, Model, ModelZoo
from celonis_ml.trainers import trainers_utils

LOG_LEVELS = {
    "FATAL": logging.FATAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARN,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


class CelonisMLTrainer:
    """Generic Celonis Trainer class.
    """

    def __init__(self, data_dir, cel_analysis, test_size=0.1, valid_size=0.1, verbosity="DEBUG", **kwargs):

        """Generic constructor of a trainer.
        """
        self.data_dir = Path(data_dir)
        self.training_output_dir = os.path.join(self.data_dir, "training")
        self.prediction_output_dir = os.path.join(self.data_dir, "prediction")
        self.cel_analysis = cel_analysis
        self.test_size = test_size
        self.valid_size = valid_size
        self.trained_models = set()
        self._classes = None
        self._zoo_models = {}

        # general information
        self.verbosity = LOG_LEVELS.get(verbosity, logging.INFO)
        self._logger = logging.getLogger(__name__)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def train(self, model_name, **kwargs):
        """Generic train function.
        """
        self.data_init = DataLoader(
            purpose="training", data_dir=self.data_dir, cel_analysis=self.cel_analysis, **kwargs
        )

        X_train, y_train = self.data_init.obtain_data(cases_only=True, **kwargs)

        if model_name is None or isinstance(model_name, list):
            model_output_path = self.training_output_dir
            results, feature_importance = self._train_zoo(
                X_train, y_train, model_name, output_path=model_output_path, **kwargs
            )
        if isinstance(model_name, str):
            model_output_file = os.path.join(self.training_output_dir, model_name + ".mod")
            results, feature_importance = self._train_one(X_train, y_train, model_name, model_output_file, **kwargs)

        results["_CASE_KEY"] = results["_CASE_KEY"].astype(str)
        dump_data_file(os.path.join(self.training_output_dir, "training_results.pkl"), results)
        dump_data_file(os.path.join(self.training_output_dir, "class_labels.pkl"), self._classes)

        test_performance = None
        for m in self.trained_models:
            subresults = results[(results.usage == "test") & (results.model_name == m)]

            if self._zoo_models[m].type == "regressor":
                tmp = performance_regression(subresults["y_true"], subresults["y_pred"]).iloc[0]
            else:
                pred_names = ["y_pred", "class"] if self._classes is not None else ["y_pred"]
                tmp = performance_classification(subresults["y_true"], subresults[pred_names], 0.5, self._classes)

            tmp["model_name"] = [m] * tmp.shape[0]
            test_performance = test_performance.append(tmp, ignore_index=True) if test_performance is not None else tmp

        self._logger.info(f"Training results header:\n{results.head(10)}")
        self._logger.info(f"Performance on test set:\n{test_performance}")

        return results, test_performance, feature_importance

    def _train_one(self, X_train, y_train, model_name, model_output_file, save_dataset=True, params=None, **kwargs):

        """Takes the training data extracted from Celonis and trains the models specified in the Trainer instance. The
        trained models are saved to be used for predictions. Doesn't support scipy sparse matrices yet.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Training dataset. From this, validation and test sets will be extracted if necessary. If the chosen model
            accepts it, a test set will be used in the fit function.

        y_train : numpy.ndarray
            Target variable. Might be discrete or continuous values depending on the task.

        model_name : str
            One of the models available at `models_zoo`. Depending on the choice, the task will be that of
            classification or regression.

        model_output_file:
            Path where the trained model should be saved.

        Returns
        -------
        pandas.DataFrame
            Training results after applying the corresponding model. This dataframe contains the `y_pred`, `y_true` and 
            `usage` for each row (training, validation, testing).

        """
        zoo_model = ModelZoo.get_model_class(model_name)()
        if save_dataset:
            pth = model_output_file.split(os.path.basename(model_output_file))[0]
            dump_data_file(os.path.join(pth, "X_train.parquet"), X_train)
            dump_data_file(os.path.join(pth, "y_train.parquet"), y_train)

        # Sample a portion of the training set for testing purposes
        if zoo_model.type == "regressor":
            splits_1 = model_selection.train_test_split(X_train, y_train, test_size=(self.test_size), random_state=111)
        else:
            splits_1 = model_selection.train_test_split(
                X_train, y_train, test_size=(self.test_size), random_state=111, stratify=y_train
            )

        X_train, X_test, y_train, y_test = splits_1

        self._X_train = X_train.astype(np.float32)
        self._X_test = X_test.astype(np.float32)

        self._y_train = y_train
        self._y_test = y_test

        train = zoo_model.set_training_data(self._X_train, self._y_train)
        zoo_model.initialize()

        if zoo_model.type == "classifier":
            train, tmp_classes = train
            self._classes = (
                tmp_classes
                if (self._classes is None) or (tmp_classes is None)
                else (self._classes if len(self._classes) > len(tmp_classes) else tmp_classes)
            )

        if kwargs.get("tuning"):
            best_parameters = zoo_model.optimize_model(**kwargs)
            zoo_model.fit(train, best_parameters)
        else:
            zoo_model.fit(train, params)

        feature_importance = zoo_model.get_feature_importance()

        # Store model for later prediction
        zoo_model.save_model(model_output_file)
        self._zoo_models[model_name] = zoo_model

        y_pred_train = zoo_model.predict(self._X_train)
        y_pred_test = zoo_model.predict(self._X_test)

        # Storing the training results for reporting in Celonis
        if self._classes is not None:
            # Multi-class classification
            training_results = {"_CASE_KEY": [], "usage": [], "y_true": [], "y_pred": [], "class": [], "model_name": []}
            for i, cname in enumerate(self._classes):
                for y_true, y_pred, x, usage in zip(
                    [y_train, y_test], [y_pred_train, y_pred_test], [X_train, X_test], ["train", "test"]
                ):
                    l = x.shape[0]
                    training_results["_CASE_KEY"] += list(x.index.values)
                    training_results["usage"] += [usage] * l
                    training_results["y_true"] += list(y_true)
                    training_results["y_pred"] += list(y_pred[:, i])
                    training_results["class"] += [cname] * l
                    training_results["model_name"] += [model_name] * l
        else:
            # Binary or regression
            training_results = {"_CASE_KEY": [], "usage": [], "y_true": [], "y_pred": [], "model_name": []}
            for y_true, y_pred, x, usage in zip(
                [y_train, y_test], [y_pred_train, y_pred_test], [X_train, X_test], ["train", "test"]
            ):
                l = x.shape[0]
                training_results["_CASE_KEY"] += list(x.index.values)
                training_results["usage"] += [usage] * l
                training_results["y_true"] += list(y_true)
                training_results["y_pred"] += list(y_pred)
                training_results["model_name"] += [model_name] * l

        training_results = pd.DataFrame.from_dict(data=training_results)
        self.trained_models.add(model_name)

        return training_results, feature_importance

    def _train_zoo(self, X_train, y_train, model_name, output_path=None, **kwargs):

        """Trains all models specified in the list provided on `model_name` and collects the general results. Also
        states the best classifier for the given task and dataset.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Training dataset.

        y_train : numpy.ndarray
            Target variable. Might be discrete or continuous values depending on the task.

        model_name : list of str
            List of strings corresponding to models available at `models_zoo`. If empty, the training will be performed
            on all `AVAILABLE_MODELS`.

        Returns
        -------
        pandas.DataFrame
            results of the best performing model and statistics of the rest, after training all models available in the
            model zoo.
        """
        if isinstance(model_name, list):
            iterator = model_name
        else:
            iterator = AVAILABLE_MODELS

        training_results = None
        feature_importance_results = None
        for model_name in iterator:
            self._logger.info("Training the model: " + model_name + "...")
            model_instance = ModelZoo.get_model_class(model_name)
            self.model = model_instance()

            model_output_file = os.path.join(output_path, model_name + ".mod")
            res, partial_feature_importance = self._train_one(X_train, y_train, model_name, model_output_file, **kwargs)

            if training_results is None:
                training_results = res
            else:
                training_results = pd.concat([training_results, res])

            if feature_importance_results is None:
                feature_importance_results = partial_feature_importance
            else:
                feature_importance_results = pd.concat([feature_importance_results, partial_feature_importance])

        # TODO: Add model benchmarking stats
        return training_results, feature_importance_results

    def _predict_one(self, X_pred, model_name, model_path, output_path=None, verify_set=True, **kwargs):

        """Gets predictions for the model specified in the `model_name`. The training must have already been ran in
        order to predict.

        Parameters
        ----------
        X_pred : pandas.DataFrame
            Dataset to be predicted for.

        model_name : str
            Name of the model to be used for prediction. This model should be available at `models_zoo`. Also, it must
            exist at `model_path` otherwise, prediction is not possible.

        model_path : str
            Location where the trained model will be searched. If the path is invalid, a FileNotFoundError is thrown.

        output_path : str
            Location where the prediction dataset should be persisted. Will be deprecated in a future version.

        Returns
        -------
        pandas.DataFrame
            `y_pred` results obtained from `model_name` on a case level.
        """
        zoo_model = ModelZoo.load_model(model_name, model_path)

        if verify_set:
            X_train_path = os.path.join(output_path.replace("/prediction/", "/training/"), "X_train.parquet")
            X_pred_path = os.path.join(output_path, "X_pred.parquet")
            self._X_train = load_data_file(X_train_path)

            X_pred = verify_pred_set(self._X_train, X_pred, X_pred_path, self._logger, X_train_path=X_train_path)

        X_sp = sparse.csc_matrix(X_pred, dtype=np.float32)
        y_pred = zoo_model.predict(X_sp)

        if len(y_pred.shape) > 1:
            pre = {"_CASE_KEY": [], "y_pred": [], "class": [], "model_name": []}
            for i, cname in enumerate(self._classes):
                l = X_pred.shape[0]
                pre["_CASE_KEY"] += list(X_pred.index.values)
                pre["y_pred"] += list(y_pred[:, i])
                pre["class"] += [cname] * l
                pre["model_name"] += [model_name] * l
                predictions = pd.DataFrame.from_dict(pre)
        else:
            predictions = pd.DataFrame(
                {"_CASE_KEY": X_pred.index, "y_pred": y_pred, "model_name": [model_name] * len(y_pred)}
            )
        predictions["_CASE_KEY"] = predictions["_CASE_KEY"].astype("str")

        return predictions

    def _predict_zoo(self, X_pred, model_list, model_path, output_path, **kwargs):

        """Gets predictions for all models specified in the `model_list` and collects the general results. The training
        must have already been ran in order to predict.

        Parameters
        ----------
        X_pred : pandas.DataFrame
            Dataset to be predicted for.

        model_list : list of str
            List of strings corresponding to models available at `models_zoo`. Also, these models must exist at
            `model_path` otherwise, prediction is not possible.

        model_path : str
            Location where the trained models will be searched. If the path is invalid, a FileNotFoundError is thrown.

        output_path : str
            Location where the prediction dataset should be persisted. Will be deprecated in a future version.

        Returns
        -------
        pandas.DataFrame
            `y_pred` results for each model in `model_list`.
        """
        predictions = None
        # TODO: Make case key independent. Will we ever want to train a dataset that doesn't use a case key as index?
        for m in model_list:
            mp = os.path.join(model_path, m + ".mod")
            y_pred = self._predict_one(X_pred, m, mp, output_path, **kwargs)
            if predictions is None:
                predictions = y_pred
            else:
                predictions = pd.concat([predictions, y_pred])

        predictions["_CASE_KEY"] = predictions["_CASE_KEY"].astype("str")
        return predictions

    def predict(self, model_name=None, **kwargs):
        """
        Gathers the data of the open cases and the models trained during the last training and performs classification
        or regression depending on the chosen model. If push == True, the results are pushed back to the Celonis Data
        Model.

        Returns
        --------
        predictions : pandas DataFrame
            Predictions to be pushed back to Celonis,
            containing the following columns:

            - _CASE_KEY : unique identifier for each Celonis case.
            - y_pred{_model}: predicted target extracted from the selected model.

        """
        self.data_init = DataLoader(
            purpose="prediction", data_dir=self.data_dir, cel_analysis=self.cel_analysis, **kwargs
        )

        self._logger.info("Start predicting...")

        # Obtain data from Celonis
        X_pred, _ = self.data_init.obtain_data(cases_only=True, **kwargs)
        assert X_pred is not None and X_pred.shape[0] > 0, "No cases found."

        # Retrieve class labels
        self._classes = load_data_file(os.path.join(self.training_output_dir, "class_labels.pkl"))

        # Warning!!: This is only taking by reference the models in the first reaction time folder
        trained_models = glob.glob(str(self.training_output_dir) + "/*.mod")
        assert len(trained_models) > 0, (
            "No trained models found in the given directory. " "You must train at least one model before predicting."
        )
        trained_models = [os.path.basename(f).split(".mod")[0] for f in trained_models]

        if isinstance(model_name, list):
            trained_models = model_name
            model_name = None

        prediction_output = self.training_output_dir

        if model_name is None:
            predictions = self._predict_zoo(
                X_pred, trained_models, self.training_output_dir, prediction_output, **kwargs
            )
        else:
            model_path = os.path.join(self.training_output_dir, model_name + ".mod")
            predictions = self._predict_one(X_pred, model_name, model_path, prediction_output, **kwargs)

        predictions.to_pickle(os.path.join(prediction_output, "y_pred.parquet"))
        self._logger.info(f"Prediction results header:\n{predictions.head(10)}")
        return predictions


class TimeTranchesTrainer(CelonisMLTrainer):

    """
    CelonisMLTrainer with the Time Tranches modality. This trainer is intended to be used for the On-Time Delivery
    use-case, on Purchase-to-Pay  or 0rder-to-Cash processes.
    
    With this methodology, the intention is to replicate the state of a process at a certain point in time. For example,
    if we want to predict something for a case that will be due in N days, we should use a model that has been trained
    on cases at the point in time when they had N days to due date. Like this, we get a glimpse of how things have
    looked like at point N in the past, in order to make predictions for open cases.

    Attributes
    ----------
    data_dir : path_like
        base folder where to store the data.
    cel_analysis: pycelonis.BaseAnalysis
        Celonis Analysis object to exchange data.
    reaction_time : numeric or list of numeric
        Number of days between `TODAY` and the due date of the cases to be analysed.
    test_size : float
        Number between 0 and 1 indicating the percentaje of rows to be extracted from the training set to generate the
        test set.
    valid_size : float
        Number between 0 and 1 indicating the percentaje of rows to be extracted from the training set to generate the
        validation set.
   verbosity : {'FATAL', 'ERROR', 'WARN', 'INFO', 'DEBUG'}
        print status updates as the data is processed. Equivalent to setting a log level.
    """

    def __init__(
        self,
        data_dir,
        cel_analysis,
        reaction_time=None,
        shared_selection_url=None,
        test_size=0.1,
        valid_size=0.1,
        verbosity="DEBUG",
        **kwargs,
    ):

        CelonisMLTrainer.__init__(self, data_dir, cel_analysis, test_size, valid_size, verbosity, **kwargs)
        self.reaction_time = [reaction_time] if isinstance(reaction_time, int) else reaction_time
        self.shared_selection_url = shared_selection_url

    def _get_dataset_for_step(self, **kwargs):
        """This function extracts cases with number of days between `today` and `due_date` equals to `reaction_time`.
        If no reaction time has been selected, all cases are returned.
        """

        X, y_train = self.data_init.obtain_data(**kwargs)
        return X, y_train

    def _get_trained_models(self, model_name=None):
        if model_name:
            model_name = [model_name] if isinstance(model_name, str) else model_name
            trained_models = {m: [] for m in model_name}
        for rt in self.reaction_time:
            models_here = glob.glob(str(os.path.join(self.training_output_dir, str(rt))) + "/*.mod")
            models_here = [os.path.basename(f).split(".mod")[0] for f in models_here]
            if len(models_here) != 0:
                if model_name:
                    for m in model_name:
                        if m in models_here:
                            trained_models[m] = rt
                else:
                    trained_models[m] = rt

        assert len(trained_models) > 0, (
            "No trained models found in the given directories. " "You must train at least one model before predicting."
        )

        return trained_models

    def train(self, model_name=None, save_dataset=True, params=None, **kwargs):
        """
        Performs training with the Time Tranches modality on one or several models. The model(s) must exist in the
        `AVAILABLE_MODELS` dictionary. A `DataLoader` object to trigger the data extraction, the specific dataset for
        each `reaction_time` is obtained with `_get_dataset_for_step`.

        Returns
        -------
        pandas.DataFrame
            results after training the selected model using the corresponding dataset for each reaction_time.
        """

        self.data_init = DataLoader(
            purpose="training",
            data_dir=self.data_dir,
            cel_analysis=self.cel_analysis,
            shared_selection_url=self.shared_selection_url,
            **kwargs,
        )

        # TODO do reaction time in 1 place
        if self.reaction_time is None:
            v = self.data_init._variables_filters_query.variables.get("reaction_time")
            if v:
                self.reaction_time = [int(r) for r in v.split(",")]

        # Create the output path
        for rt in self.reaction_time:
            create_path(os.path.join(self.training_output_dir, str(rt)))

        results = None
        feature_importance = None
        if model_name is None or isinstance(model_name, list):
            # Run predictions for each day in the reaction time window
            for rt in self.reaction_time:
                self._logger.info(f"Training at {rt} days to due date.")
                X_train, y_train = self._get_dataset_for_step(reaction_time=rt, **kwargs)

                model_output_path = os.path.join(self.training_output_dir, str(rt))
                partial_results, partial_feature_importance = self._train_zoo(
                    X_train, y_train, model_name, output_path=model_output_path, **kwargs
                )
                partial_results["wildcard"] = [rt] * partial_results.shape[0]
                partial_feature_importance["wildcard"] = [rt] * partial_feature_importance.shape[0]
                results = partial_results if results is None else pd.concat([results, partial_results])
                feature_importance = (
                    partial_feature_importance
                    if feature_importance is None
                    else pd.concat([feature_importance, partial_feature_importance])
                )

        if isinstance(model_name, str):
            for rt in self.reaction_time:
                self._logger.info(f"Training at {rt} days to due date.")
                X_train, y_train = self._get_dataset_for_step(reaction_time=rt, **kwargs)

                model_output_path = os.path.join(self.training_output_dir, str(rt))
                model_output_file = os.path.join(model_output_path, model_name + ".mod")
                preds_file = os.path.join(model_output_path, model_name + ".pkl")

                partial_results, partial_feature_importance = self._train_one(
                    X_train, y_train, model_name, model_output_file, preds_file, params=params, **kwargs
                )
                # In this use case wildcard is reaction_time
                partial_results["wildcard"] = [rt] * partial_results.shape[0]
                partial_feature_importance["wildcard"] = [rt] * partial_feature_importance.shape[0]
                results = partial_results if results is None else pd.concat([results, partial_results])
                feature_importance = (
                    partial_feature_importance
                    if feature_importance is None
                    else pd.concat([feature_importance, partial_feature_importance])
                )

        results["_CASE_KEY"] = results["_CASE_KEY"].astype(str)
        dump_data_file(os.path.join(self.training_output_dir, "training_results.pkl"), results)
        dump_data_file(os.path.join(self.training_output_dir, "class_labels.pkl"), self._classes)

        test_performance = None

        for rt in self.reaction_time:
            subresults = results[(results.usage == "test") & (results.wildcard == rt)]
            for m in self.trained_models:
                subresults = results[(results.usage == "test") & (results.wildcard == rt) & (results.model_name == m)]

                if self._zoo_models[m].type == "regressor":
                    tmp = performance_regression(subresults["y_true"], subresults["y_pred"]).iloc[0]
                else:
                    pred_names = ["y_pred", "class"] if self._classes is not None else ["y_pred"]
                    tmp = performance_classification(subresults["y_true"], subresults[pred_names], 0.5, self._classes)

                tmp["wildcard"], tmp["model_name"] = [rt] * tmp.shape[0], [m] * tmp.shape[0]
                test_performance = (
                    test_performance.append(tmp, ignore_index=True) if test_performance is not None else tmp
                )

        self._logger.info(f"Training results header:\n{results.head(10)}")
        self._logger.info(f"Performance on test set:\n{test_performance}")

        return results, test_performance, feature_importance

    def predict(self, model_name=None, **kwargs):
        """
        Specific prediction method in the Time Tranches modality. Gathers the data of the open cases and the models
        trained during the last training and performs classification or regression depending on the chosen model. If
        push == True, the results are pushed back to the Celonis Data Model.

        Returns
        --------
        predictions : pandas DataFrame
            Predictions to be pushed back to Celonis,
            containing the following columns:

            - _CASE_KEY : unique identifier for each Celonis case.
            - y_pred{_model}: predicted target extracted from the selected model.

        """

        self.data_init = DataLoader(
            purpose="prediction",
            data_dir=self.data_dir,
            cel_analysis=self.cel_analysis,
            shared_selection_url=self.shared_selection_url,
            **kwargs,
        )

        # TODO do reaction time in 1 place
        if self.reaction_time is None:
            v = self.data_init._variables_filters_query.variables.get("reaction_time")
            if v:
                self.reaction_time = [int(r) for r in v.split(",")]

        # Create the output path
        for rt in self.reaction_time:
            create_path(os.path.join(self.prediction_output_dir, str(rt)))

        # Retrieve class labels
        self._classes = load_data_file(os.path.join(self.training_output_dir, "class_labels.pkl"))

        # Warning!!: This is only taking by reference the models in the first reaction time folder
        trained_models = glob.glob(str(os.path.join(self.training_output_dir, str(self.reaction_time[0]))) + "/*.mod")
        assert len(trained_models) > 0, (
            "No trained models found in the given directory. " "You must train at least one model before predicting."
        )
        trained_models = [os.path.basename(f).split(".mod")[0] for f in trained_models]

        if isinstance(model_name, list):
            trained_models = model_name
            model_name = None

        predictions = None
        if model_name is None:
            for rt in self.reaction_time:
                self._logger.info(f"Predicting at {rt} days to due date.")

                # Obtain data from Celonis
                X_pred, _ = self.data_init.obtain_data(reaction_time=rt)
                assert X_pred is not None and X_pred.shape[0] > 0, "No cases found."

                model_path = os.path.join(self.training_output_dir, str(rt))
                po = os.path.join(self.prediction_output_dir, str(rt))
                preds = self._predict_zoo(X_pred, trained_models, model_path, po, **kwargs)
                preds["wildcard"] = [rt] * preds.shape[0]
                predictions = preds if predictions is None else pd.concat([predictions, preds])
        else:
            for rt in self.reaction_time:
                self._logger.info(f"Predicting at {rt} days to due date.")

                # Obtain data from Celonis
                try:
                    X_pred, _ = self.data_init.obtain_data(reaction_time=rt)
                except Exception as e:
                    self._logger.warn(f"Error getting data for {rt} days to due date. Skipping. The error was {e}.")
                    continue

                model_path = os.path.join(self.training_output_dir, str(rt), model_name + ".mod")
                po = os.path.join(self.prediction_output_dir, str(rt))
                preds = self._predict_one(X_pred, model_name, model_path, po, **kwargs)
                preds["wildcard"] = [rt] * preds.shape[0]
                predictions = preds if predictions is None else pd.concat([predictions, preds])

        self._logger.info(f"Prediction results header:\n{predictions.head(10)}")
        return predictions
