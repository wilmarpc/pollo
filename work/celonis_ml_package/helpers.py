import calendar
import csv
import logging
import os
import pickle
import shutil
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tqdm import tqdm

logger = logging.getLogger(__name__)


def set_log_level(log_level):
    import logging

    # Set up logger
    if log_level != logging.INFO:
        logger = logging.getLogger()
        logger.setLevel(log_level)
        for handler in logger.handlers:
            handler.setLevel(log_level)


def get_data_dir():
    """Returns the data directory
    """
    import inspect

    filename = Path(inspect.getframeinfo(inspect.currentframe()).filename)
    this_path = filename.absolute().parents[1]
    DATA = this_path / "example" / "data"
    create_path(DATA)
    return DATA.absolute()


def get_example_dir():
    """Returns the example directory
    """
    import inspect

    filename = Path(inspect.getframeinfo(inspect.currentframe()).filename)
    this_path = filename.absolute().parents[1]
    DATA = this_path / "example"
    create_path(DATA)
    return DATA.absolute()


def get_externals_dir():
    """Returns the example directory
    """
    import inspect

    filename = Path(inspect.getframeinfo(inspect.currentframe()).filename)
    this_path = filename.absolute().parents[0]
    DATA = this_path / "externals"
    create_path(DATA)
    return DATA.absolute()


def get_docs_dir():
    """Returns the docs directory
    """
    import inspect

    filename = Path(inspect.getframeinfo(inspect.currentframe()).filename)
    this_path = filename.absolute().parents[1]
    DATA = this_path / "docs"
    create_path(DATA)
    return DATA.absolute()


def get_projects_dir():
    """Returns the projects directory
    """
    import inspect

    filename = Path(inspect.getframeinfo(inspect.currentframe()).filename)
    this_path = filename.absolute().parents[0]
    DATA = this_path / "projects"
    create_path(DATA)
    return DATA.absolute()


def get_analyses_dir():
    """Returns the analyses directory
    """
    import inspect
    from pathlib import Path

    filename = Path(inspect.getframeinfo(inspect.currentframe()).filename)
    this_path = filename.absolute().parents[0]
    DATA = this_path / "templates" / "analyses"
    create_path(DATA)
    return DATA.absolute()


def get_demo_notebooks_dir():
    """Returns the demo notebooks directory
    """
    import inspect
    from pathlib import Path

    filename = Path(inspect.getframeinfo(inspect.currentframe()).filename)
    this_path = filename.absolute().parents[0]
    DATA = this_path / "templates" / "notebooks"
    create_path(DATA)
    return DATA.absolute()


def create_path(directory):
    """
    Creates a directory in the given path, if it doesn't already exist.

    Parameter
    ---------
    directory : path_like
        path to the directory to be created

    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def delete_path(directory):
    """
    Deletes recursively the directories in the given path, if it exists.

    Parameter
    ---------
    directory : path_like
        path to the directory to be deleted

    """
    if os.path.exists(directory):
        shutil.rmtree(directory)


def extract_helping_features(event_log, case_log):
    """
    Computes features based on case and event log which are necessary for
    preprocessing (removing incomplete cases) and labeling the data frames, 
    such as throughput times and indices of critical events.

    Parameters
    ----------
    event_log : pandas.DataFrame
        dataframe with a log of events.
    case_log : pandas.DataFrame
        dataframe with a log of cases.

    Returns
    -------
    pandas.DataFrame
        inputted event log and with features enriched case
        log.

    """
    event_log_grouped = event_log.groupby("_CASE_KEY")

    # extract basic help features
    case_log_features = event_log_grouped.agg({"Activity_EN": {"EVENT_COUNT_TOTAL": "count"}})
    case_log_features.columns = case_log_features.columns.droplevel(0)
    case_log_features = case_log_features.reset_index()
    case_log_enriched = case_log_features.merge(case_log, how="left", on=["_CASE_KEY"])

    return case_log_enriched


def expand_time(df):
    """
    Expanding the time feature to year, month, day, hour, weekday, and the gap
    between the current activity and the one before.

    Parameters
    ----------
    event_log_compl : pandas.DataFrame
        event log.

    Returns
    -------
    pandas.DataFrame
        extracted/added features.

    """
    # Make sure column is datetime format
    df.loc[:, "EVENTTIME"] = pd.to_datetime(df["EVENTTIME"], unit="ms")

    df["YEAR"] = df["EVENTTIME"].apply(lambda x: int(x.year))
    df["MONTH"] = df["EVENTTIME"].apply(lambda x: int(x.month))
    df["DAY"] = df["EVENTTIME"].apply(lambda x: int(x.day))
    df["HOUR"] = df["EVENTTIME"].apply(lambda x: int(x.hour))
    df["WEEKDAY"] = df["EVENTTIME"].apply(lambda x: calendar.day_name[x.weekday()])
    tmp = df.groupby("_CASE_KEY")["EVENTTIME"].diff().fillna(0)
    df["GAP"] = tmp.dt.total_seconds()
    df.drop(["EVENTTIME"], axis=1, inplace=True)
    return df


# Encoding (index_based_encoding)


def index_based_encode(event_log, column, max_events):
    """
    Converts the respective column in the by case key-grouped DataFrame to new
    columns, one for each value up to the maximum number of events. The
    resulting data frame contains one row per case and one column for each
    column value between 0 and max_events.

    Parameters
    ----------
    event_log : pandas.DataFrame
        event log.
    column : str
        Column that should be spread into new columns.
    max_events : int
        Maximum numer of values that should be considered.
    
    Returns
    -------
    pandas.DataFrame
        New DataFrame with column values as new columns (per
        _CASE_KEY).

    """
    event_log_grouped = event_log.groupby("_CASE_KEY")

    # Generate new column names.
    cols_index = column + "_" + pd.Index(list(range(1, max_events + 1))).astype("str")
    # select control flow until maximum amount of events
    temp_agg = event_log_grouped.agg({column: {"TEMP": (lambda x: tuple(x)[0:max_events])}})
    temp_agg.columns = temp_agg.columns.droplevel(0)
    temp_agg = temp_agg.reset_index()

    # spread event selection over columns
    temp = temp_agg["TEMP"].apply(pd.Series)
    temp.columns = cols_index

    temp = pd.concat([pd.DataFrame(temp_agg["_CASE_KEY"]), temp], axis=1)
    return temp


def representsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def remove_features(case_log, step, max_events):
    """
    Removing unncessary features based on the current time step.

    Parameters
    ----------
    case_log : pandas.DataFrame
        encoded case log (index or frequency based one + one
        hot encoding after those).
    step : int
        current time step.
    max_events : int
        maximum number of events for this step.
    
    Returns
    -------
    pandas.DataFrame
        input case log excluding the with unncessary/removed
        features.

    """
    next_activities_to_drop = []
    f_steps = [int(i) for i in range(step + 1, max_events + 1)]
    for c in case_log.columns:
        if len(c.split("_")) > 1:
            index = c.split("_")[-2]
            if representsInt(index):
                index = int(index)
            else:
                index = c.split("_")[-1]
                if representsInt(index):
                    index = int(index)
                else:
                    continue
            if any(i == index for i in f_steps):  # seems suspicious
                next_activities_to_drop = next_activities_to_drop + [c]

    X_drop_columns = next_activities_to_drop + ["EVENT_COUNT_TOTAL"]
    new_case_log = case_log.drop(X_drop_columns, axis=1)

    cat_indices = np.where(new_case_log.dtypes != np.float)[0]
    return new_case_log, None, cat_indices


def confusion_matrix_at_threshold(y_true, y_pred_proba, thre, performance, step, y_train):
    """
    Prints the confusion matrix and calculates performance for the current
    step.d

    Parameters
    ----------
    y_true : pandas.Series
        actual class label.
    y_pred_proba : pandas.Series
        predicted class.
    thre : float
        threshold
    performance : pandas.DataFrame
        performance of the model until now.
    step : int
        training step.
    y_train : pandas.Series
        actual class label of the training set.
    
    Returns
    -------
    pandas.DataFrame
        The initial performance dataframe plus the performance
        of the current step.

    """
    y_pred = (y_pred_proba > thre) * 1
    print("In the test dataset:")
    print("Negative class: " + str(np.sum(y_true == 0)))
    print("Positive class: " + str(np.sum(y_true)))
    # print("ROC AUC score:", roc_auc_score(y_true,y_pred))
    print(confusion_matrix(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred).ravel()
    if len(cm) == 4:
        tn, fp, fn, tp = cm
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        performance = performance.append(
            {
                "number of activities": step,
                "train size": len(y_train),
                "test size": len(y_true),
                "accuracy": accuracy_score(y_true, y_pred),
                "true delayed predicted": tp,
                "false delayed predicted": fp,
                "false on-time predicted": fn,
                "true on-time predicted": tn,
                "precision score": precision_score(y_true, y_pred),
                "recall score": recall_score(y_true, y_pred),
                "roc auc score": roc_auc_score(y_true, y_pred),
                "f1 score": f1_score(y_true, y_pred),
            },
            ignore_index=True,
        )
    elif len(cm) == 1:
        print("All true and predicted labels are equal.")
        performance = cm

    return performance


def coerce_series_to_datetime(series, format="%Y-%m-%dT%H:%M:%S"):
    res = pd.to_datetime(series, format=format, errors="coerce")
    return res


def best_cutoff(y_train, y_pred_prob_train):
    """
    Calculates best cutoff in the prediction from which examples
    are deemed positive or negative. This is intended for the 
    binary classification task only. It extracts the threshold that
    minimizes the following penalized Euclidean distance:
    .. math:: d = math.sqrt{x^2+(1-y)^2}

    Parameters
    ----------
    y_train : pandas.Series
        actual class of the training set.
    y_pred_prob_train : pandas.Series
        predicted class probability.
    
    Returns
    -------
    float
        Calculated best cutoff according to penalized
        Euclidean distance.

    """

    def score(y_train, y_pred_prob_train, t):

        score = 0.2 * precision_score(y_train, np.where(y_pred_prob_train > t, 1, 0)) + 0.8 * f1_score(
            y_train, np.where(y_pred_prob_train > t, 1, 0)
        )
        return score

    best_score = score(y_train, y_pred_prob_train, 0.5)
    best = 0.5
    for t in np.arange(0.5, 0.1, -0.01):
        score_t = score(y_train, y_pred_prob_train, t)
        if best_score < score_t:
            best = t
            best_score = score_t
    print(("best cutoff is : %.2f" % best))

    return best


def save_results(db, step, dest):
    """
    Writing the results from memory into an excel file per time step and
    encoding type.

    Parameters
    ----------
    step : int
        processing step.
    dest : path_like
        path where results are to be saved.

    """
    classsifier = ["classifier"]
    training_cols = ["AUC_training"]
    test_cols = ["AUC_test"]
    time = ["execution time (minutes)"]
    Best_classifier = ["best_classifier"]

    header = classsifier + training_cols + test_cols + time + Best_classifier

    with open(dest + str(step) + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in db:
            w.writerow(row)
    f.close()


def custom_asymmetric_train(y_true, y_pred):
    """
    Custom loss function for training binary classifiers with 
    unbalanced data.
    # TODO: Add a better explanation to this.

    Parameters
    ----------
    y_true : pandas.Series
        actual class labels of the training set.
    y_pred : pandas.Series
        predicted class labels of the training set.
    
    Returns
    -------
    grad : float
        Gradient.
    hess : float
        Hessian.

    """
    residual = (np.array(y_true) - np.array(y_pred)).astype("float32")
    grad = np.where(residual < 0, -2 * 10.0 * residual, -2 * residual)
    hess = np.where(residual < 0, 2 * 10.0, 2.0)
    return (grad, hess)


def custom_asymmetric_valid(y_true, y_pred):
    """
    Custom loss function for validating binary classifiers with 
    unbalanced data.
    # TODO: Add a better explanation to this.

    Parameters
    ----------
    y_true : pandas.Series
        actual class labels of the validation set.
    y_pred : pandas.Series
        predicted class labels of the validation set.
    
    Returns
    -------
    grad : float
        Gradient.
    hess : float
        Hessian.

    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.label
    residual = (np.array(y_true) - np.array(y_pred)).astype("float")
    loss = np.where(residual < 0, (residual ** 2) * 10.0, residual ** 2)
    return ("custom_asymmetric_eval", np.mean(loss), False)


def grid_search_cv(X, y, estimator, param_grid, scoring, nfolds=2, refit="AUC", n_jobs=-1):

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=nfolds,
        refit=refit,
        n_jobs=n_jobs,
        pre_dispatch=2,
    )
    grid_search.fit(X, y)

    print("Best Score:")
    print(grid_search.best_score_)

    print("Best params:")
    print(grid_search.best_params_)

    return grid_search


def random_search_cv(X, y, estimator, param_grid, scoring, niter=2, refit="AUC", n_jobs=-1):
    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        scoring=scoring,
        n_iter=niter,
        refit=refit,
        n_jobs=n_jobs,
        pre_dispatch=2,
    )
    random_search.fit(X, y)

    print("Best Score:")
    print(random_search.best_score_)

    print("Best params:")
    print(random_search.best_params_)

    return random_search


def best_threshold_for_f1(dataset, true_col, pred_col, name):
    test_labels, predictions = dataset[true_col], dataset[pred_col]
    precision, recall, thresholds = precision_recall_curve(test_labels, predictions)
    thresholds = np.append(thresholds, 1)
    F1, accuracy = [], []
    for i in range(len(thresholds)):
        F1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]))

    mini_thresholds = np.arange(0.3, 1, 0.001)
    for t in tqdm(mini_thresholds):
        y_pred_label = [1 if p >= t else 0 for p in predictions]
        accuracy.append(accuracy_score(test_labels, y_pred_label))
    print(f"Maximum F1 score in {name} set is at threshold {thresholds[np.argmax(F1)]}")
    print(f"Maximum Accuracy score in {name} set is at threshold {mini_thresholds[np.argmax(accuracy)]}")
    return precision, recall, thresholds


def performance_classification(y_true, y_pred, thres=0.5, labels=None):
    from sklearn.metrics import classification_report

    if y_pred.shape[1] > 1:
        # Multi-class classification
        # Old version: y_pred_labels = [labels[np.argmax(x)] for x in y_pred.values]
        y_pred_labels = y_pred.reset_index().groupby("index").max()["class"]
        y_true = y_true.reset_index().drop_duplicates().set_index("index")

        report = classification_report(y_true, y_pred_labels, labels=labels, output_dict=True)
        df, df2 = pd.DataFrame(), pd.DataFrame()
        for c in labels:
            df = pd.concat([df, pd.DataFrame(report[c], index=[c])])
        df = df.reset_index().rename(columns={"index": "class"})

        for a in [x for x in report.keys() if "avg" in x]:
            df2 = pd.concat([df2, pd.DataFrame(report[a], index=[a])])
        df2 = df2.reset_index().rename(columns={"index": "averages"})
        df2["class"] = [np.nan] * df2.shape[0]

        df = pd.concat([df, df2], ignore_index=True)
        df["roc_auc"] = [np.nan] * df.shape[0]
        return df
    else:
        # Binary classification
        return performance_at_threshold(y_true, y_pred, thres)


def performance_at_threshold(y_true, y_pred_proba, thres):
    """
    Prints the confusion matrix and calculates performance for the current
    step.d

    Parameters
    ----------
    y_true : pandas.Series
        actual class label.
    y_pred_proba : pandas.Series
        predicted class.
    thres : float
        threshold
    
    Returns
    -------
    pandas.DataFrame
        The initial performance dataframe plus the performance
        of the current step.

    """
    if y_true.dtype == "object":
        cats = y_true.astype("category")
        y_true = cats.cat.codes

    if len(y_true) == 0:
        raise ValueError("Empty series.")
    performance = pd.DataFrame()
    performance["num_examples"] = [len(y_true)]
    performance["pos_cases"] = [np.sum(y_true == 1)]
    performance["neg_cases"] = [np.sum(y_true == 0)]
    performance["pos_class_ratio"] = [np.sum(y_true == 1) / len(y_true)]

    if isinstance(thres, list):
        performance["roc_auc"] = [roc_auc_score(y_true, y_pred_proba)]
        for th in thres:
            y_pred = (y_pred_proba > th) * 1
            th = str(th)
            cm = confusion_matrix(y_true, y_pred).ravel()
            if len(cm) == 4:
                tn, fp, fn, tp = cm
            else:
                tn, tp = np.sum(y_true == 0), np.sum(y_true == 1)
                fn, fp = 0, 0
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            performance["acc_" + th] = [accuracy_score(y_true, y_pred)]
            performance["tp_" + th] = [tp]
            performance["fp_" + th] = [fp]
            performance["fn_" + th] = [fn]
            performance["tn_" + th] = [tn]
            performance["preci_" + th] = [precision_score(y_true, y_pred)]
            performance["recall_" + th] = [recall_score(y_true, y_pred)]
            performance["f1_" + th] = [f1_score(y_true, y_pred)]
    else:
        try:
            performance["roc_auc"] = [roc_auc_score(y_true, y_pred_proba)]
        except:
            performance["roc_auc"] = np.nan
        y_pred = (y_pred_proba > thres) * 1
        cm = confusion_matrix(y_true, y_pred).ravel()
        if len(cm) == 4:
            tn, fp, fn, tp = cm
        else:
            tn, tp = np.sum(y_true == 0), np.sum(y_true == 1)
            fn, fp = 0, 0
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        performance["acc"] = [accuracy_score(y_true, y_pred)]
        performance["tp"] = [tp]
        performance["fp"] = [fp]
        performance["fn"] = [fn]
        performance["tn"] = [tn]
        performance["preci"] = [precision_score(y_true, y_pred)]
        performance["recall"] = [recall_score(y_true, y_pred)]
        performance["f1"] = [f1_score(y_true, y_pred)]
    return performance


def train_stacking(X, y, _data_dir):
    # TODO: Add more Level 2 Models e.g. XGBoost
    cla = LogisticRegression(random_state=0, penalty="l1", C=1e-2, n_jobs=-1)
    cla.fit(X, y)
    model_dir = _data_dir
    create_path(model_dir)
    joblib.dump(cla, os.path.join(model_dir, "stagging_log_regression.pkl"))
    return cla


def load_pickle(filename):
    with open(filename, "rb") as f:
        item = pickle.load(f)
    return item


def dump_to_pickle(filename, value):
    try:
        with open(filename, "wb") as f:
            pickle.dump(value, f)
    except OverflowError:
        logger.warn("Exporting as pickle was not possible: OverflowError. Retrying using protocol 4...")
        with open(filename, "wb") as f:
            pickle.dump(value, f, protocol=4)


def working_hours_between(
    df,
    target_col="target_eventtime",
    source_col="source_eventtime",
    workday_start=9,
    workday_end=21,
    country_code="DE",
    work_on_sunday=False,
    work_on_saturday=False,
):

    """
    Calculates the exact amount of working hours that
    pass between the source_col and the target_col of
    df, taking into coniseration working times,
    weekend and holidays.

    Parameters
    ----------
    target_activity : str
        if you only want waiting times for a single
        activity, this can be stated here

    workday_start : int
        time at which a working day starts

    workday_end : int
        time at which a working day ends

    work_on_sunday : bool
        if True sunday is reagarded as a normal working day

    work_on_saturday : bool
        if True saturday is reagarded as a normal working day

    Returns
    -------
    df : pandas.DataFrame
        the orginal dataframe df expanded with the
        column: working_hours_between

    """
    import holidays
    import numpy as np

    # change datatype to apply working days between function from np
    source = df[source_col].dt.strftime("%Y-%m-%d")
    source = np.array(source.tolist(), dtype="datetime64[D]")

    target = df[target_col].dt.strftime("%Y-%m-%d")
    target = np.array(target.tolist(), dtype="datetime64[D]")
    # add saturday, sunday if they are working days
    weekmask = [1, 1, 1, 1, 1, 0, 0]
    weekmask[5] = int(work_on_saturday)
    weekmask[6] = int(work_on_sunday)
    # adding a holiday calendar to the working day count

    # selecting the range to be checked for
    max_date = (df[target_col].max() + pd.DateOffset(years=1)).to_datetime().strftime("%Y-%m-%d")
    min_date = df[source_col].min().to_datetime().strftime("%Y-%m-%d")
    date_range = pd.date_range(start=min_date, end=max_date).strftime("%Y-%m-%d")
    # checking for holidays
    holiday_func = holidays.CountryHoliday(country_code)
    holiday_list = []
    for day in date_range.tolist():
        if day in holiday_func:
            holiday_list.append(day)
    # list of this dtype needed and used as input in busday count
    holiday_list = np.array(holiday_list, dtype="datetime64[D]")

    # hours passed on whole working days without action
    wd = np.busday_count(source, target, weekmask=weekmask, holidays=holiday_list)

    # time difference if both activities happen on the same day
    hours_between = df[target_col] - df[source_col]
    hours_between = (hours_between / np.timedelta64(1, "h")) * np.where(wd == 0, 1, 0)
    # substract 1 because target day is included
    working_hours_between = np.where(wd > 0, wd - 1, wd) * (workday_end - workday_start)
    # time elapsed at end and start of working if activites not on same day
    day1 = workday_end - df[source_col].dt.hour - df[source_col].dt.minute / 60 - df[source_col].dt.second / 3600

    day2 = df[target_col].dt.hour - workday_start + df[target_col].dt.minute / 60 + df[target_col].dt.second / 3600

    time_hours = (day1 + day2) * np.where(wd > 0, 1, 0)

    # adding up
    res = working_hours_between + time_hours + hours_between
    df["working_hours_between"] = res

    return df


def performance_regression(y_true, y_pred):
    """
    Calculates the performance metrics for the regression predictions.

    Parameters
    ----------
    y_true : pandas.Series
        actual ground truth.
    y_pred : pandas.Series
        predicted value
  
    Returns
    -------
    pandas.DataFrame
        The performance dataframe.
    """
    if len(y_true) == 0:
        raise ValueError("Empty series: y_true")
    if len(y_pred) == 0:
        raise ValueError("Empty series: y_pred")

    performance = pd.DataFrame()

    performance["explained_variance"] = [explained_variance_score(np.array(y_true), np.array(y_pred))]
    performance["r2_score"] = [r2_score(np.array(y_true), np.array(y_pred))]
    performance["mean_abs"] = [mean_absolute_error(np.array(y_true), np.array(y_pred))]
    performance["mean_sqr"] = [mean_squared_error(np.array(y_true), np.array(y_pred))]

    return performance


def plot_auc_for_result(result, system=None, plant=None, month=None, model="lgbm_classifier"):
    import matplotlib.pyplot as plt

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    if system and plant and month:
        res = result[result.System == system][result.Plant == plant][result.DUE_MONTH == month]
    else:
        res = result
    y_true = res[res.model_name == model]["y_true"].values
    y_pred = res[res.model_name == model]["y_pred"].values

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    print("ROC Score: {:.3f}".format(roc_auc_score(y_true, y_pred)))

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if system and plant and month:
        plt.title(f"ROC for {system} plant {plant} month {month}")
    else:
        plt.title(f"ROC curve")
    plt.legend(loc="lower right")
    plt.show()


def plot_feature_importances(model, data, plot_n=15, threshold=None):
    """
    Plots `plot_n` most important features and the cumulative importance of features.
    If `threshold` is provided, prints the number of features needed to reach `threshold` cumulative importance.
    Parameters
    --------
    
    plot_n : int, default = 15
        Number of most important features to plot. Defaults to 15 or the maximum number of features whichever is smaller
    
    threshold : float, between 0 and 1 default = None
        Threshold for printing information about cumulative importances

    !!!! THIS CODE HAS BEEN TAKEN AND MODIFIED FROM https://github.com/WillKoehrsen/feature-selector/blob/master/feature_selector/feature_selector.py
    """

    import matplotlib.pyplot as plt

    feature_names = data.columns
    feature_importances = pd.DataFrame({"feature": feature_names, "importance": model.feature_importance()})

    # Sort features according to importance
    feature_importances = feature_importances.sort_values("importance", ascending=False).reset_index(drop=True)

    # Normalize the feature importances to add up to one
    feature_importances["normalized_importance"] = (
        feature_importances["importance"] / feature_importances["importance"].sum()
    )
    feature_importances["cumulative_importance"] = np.cumsum(feature_importances["normalized_importance"])

    # Need to adjust number of features if greater than the features in the data
    if plot_n > feature_importances.shape[0]:
        plot_n = feature_importances.shape[0] - 1

    # Resetting plot
    plt.rcParams = plt.rcParamsDefault

    # Make a horizontal bar chart of feature importances
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # Need to reverse the index to plot most important on top
    # There might be a more efficient method to accomplish this
    axes[0].barh(
        list(reversed(list(feature_importances.index[:plot_n]))),
        feature_importances["normalized_importance"][:plot_n],
        align="center",
        edgecolor="k",
    )

    # Set the yticks and labels
    axes[0].set_yticks(list(reversed(list(feature_importances.index[:plot_n]))))
    axes[0].set_yticklabels(feature_importances["feature"][:plot_n])

    # Plot labeling
    axes[0].set_xlabel("Normalized Importance")
    axes[0].set_title("Feature Importances")

    # Cumulative importance plot
    axes[1].plot(list(range(1, len(feature_importances) + 1)), feature_importances["cumulative_importance"], "r-")
    axes[1].set_xlabel("Number of Features")
    axes[1].set_ylabel("Cumulative Importance")
    axes[1].set_title("Cumulative Feature Importance")
    plt.tight_layout()

    if threshold:
        # Index of minimum number of features needed for cumulative importance threshold
        # np.where returns the index so need to add 1 to have correct number
        importance_index = np.min(np.where(feature_importances["cumulative_importance"] > threshold))
        plt.vlines(x=importance_index + 1, ymin=0, ymax=1, linestyles="--", colors="blue")

        print("%d features required for %0.2f of cumulative importance" % (importance_index + 1, threshold))

