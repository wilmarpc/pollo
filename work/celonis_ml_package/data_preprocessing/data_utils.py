import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

_ENGINE = "pyarrow"

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    try:
        import fastparquet as fp
        import snappy

        _ENGINE = "fastparquet"
    except ImportError:
        _ENGINE = None


from celonis_ml.helpers import dump_to_pickle
from pycelonis.pql import PQL, PQLColumn, PQLFilter

from pycelonis.utils.parquet_utils import read_parquet

logger = logging.getLogger(__name__)


def balance_dataset(dataset, label_col, technique="undersampling", reduce_set=False, seed=10):
    if technique == "undersampling":
        if reduce_set:
            _, count_class_1 = dataset[label_col].value_counts()
            count_class_min = reduce_set // 2 if count_class_1 >= reduce_set // 2 else count_class_1
        else:
            count_class_min = min(dataset[label_col].value_counts())
        # Divide by class
        df_class_0 = dataset[dataset[label_col] == 0]
        df_class_1 = dataset[dataset[label_col] == 1]
        # Random under sampling
        df_class_0_under = df_class_0.sample(count_class_min, random_state=seed)
        df_class_1_under = df_class_1.sample(count_class_min, random_state=seed)

        dataset = pd.concat([df_class_0_under, df_class_1_under], axis=0)

        return dataset


def create_frame_structure(df, purpose, training_settings, signature, path_to_table, filler=0, table_name="case_table"):
    """
    Takes a df and creates a frame structure for it, so that in the
    one hot encoding, the frames have the same columns for prediction and
    training, for training loads all categories and fills up in case some
    didnt appear in the training data. For prediction takes all the columns
    from the training dataset and fills them up if they exist, else they
    are left at 0, this is for the categorical field!

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe for which the frame structure is supposed
        to be created.
    table_name : str
        if 'ct' then this is for the case table, if 'at' to
        be added.

    Returns
    -------
    pandas.DataFrame
        feature matrix with the standard structure.
    """
    # if purpose is training saves the columns and saves the df
    if purpose == "training":
        df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
        training_settings[f"{table_name}_columns_{signature}"] = df.columns.tolist()
        # Saving one-hot-encoded cases table for future runs
        dump_data_file(path_to_table, df, preserve_index=True)
        return df, training_settings

    # if purpose is prediction loads the columns and fills them with filler
    else:
        col_names_total = training_settings.get(f"{table_name}_columns_{signature}")
        if col_names_total == df.columns.tolist():
            return df, training_settings
        else:
            feature_matrix = pd.DataFrame(columns=col_names_total, index=df.index)
            for col in feature_matrix.columns.tolist():
                if col in df.columns:
                    feature_matrix[col] = df[col]
                else:
                    feature_matrix[col] = filler
        return feature_matrix, training_settings


def impute_numeric_columns(df, purpose, training_settings, imputer):
    if purpose == "training":
        training_settings[imputer] = SimpleImputer(missing_values=np.nan, strategy="median")
        training_settings[imputer].fit(df)
    return training_settings[imputer].transform(df), training_settings


def scale_numeric_columns(df, purpose, training_settings, scaler):
    if purpose == "training":
        training_settings[scaler] = RobustScaler()
        training_settings[scaler].fit(df)
    return training_settings[scaler].transform(df), training_settings


def working_hours_between(
    self,
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

    """
    # for checking
    res_df = pd.DataFrame(data = {'whole_working_days_hours':working_hours_between,
                                    "hours_star_end_of_day":time_hours,
                                    "hours_between if on same day":hours_between,
                                    "source":df["source_eventtime"].values,
                                    "target":df["target_eventtime"].values})
    """
    # adding up
    res = working_hours_between + time_hours + hours_between
    df["working_hours_between"] = res

    return df


def verify_pred_set(X_train, X_pred, X_pred_path, logger, X_train_path=None):
    X_pred = X_pred.drop(["TARGET_VARIABLE"], axis=1, errors="ignore")
    X_pred.sort_index()

    train_cols = X_train.columns

    #X_pred = X_pred.drop(["_CASE_KEY"], axis=1, errors="ignore")

    for col in list(set(train_cols).difference(set(X_pred.columns))):
        X_pred[col] = np.zeros(X_pred.shape[0])
        logger.warning("Missing Column(s) in the Prediction data set: " + col)

    X_pred = X_pred.drop(list(set(X_pred.columns).difference(set(train_cols))), axis=1)

    dump_data_file(X_pred_path, X_pred)
    return X_pred


def load_data_file(filename, from_format=None, as_object=None):
    if not from_format:
        import pathlib

        from_format = pathlib.Path(filename).suffix[1:]
    if from_format == "parquet":
        if as_object is not None and as_object != "DataFrame":
            logger.debug(
                f"Loading from {from_format} format as a {as_object} is not implemented yet. "
                "Continuing with the default set up."
            )
            as_object = "DataFrame"
        if as_object == "DataFrame" or as_object is None:
            try:
                item = read_parquet(filename)
                return item
            except:
                logger.debug(
                    f"Loading from {from_format} format as a {as_object} object failed. "
                    "Trying to load from the default format."
                )
                from_format = "pkl"
    elif from_format != "pkl":
        logger.debug(
            f"Loading {from_format} formats has not been implemented yet. " "Continuing with the default configuration."
        )
        from_format = "pkl"

    if from_format == "pkl":
        try:
            # Loading with the default, pkl
            with open(filename, "rb") as f:
                item = pickle.load(f)
            if as_object is not None and as_object not in str(type(item)):
                logger.debug(
                    f"The object was loaded with type {type(item)}. Converting to type {as_object} hasn't been "
                    "implemented. Keeping original type."
                )
            return item
        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(f"Unable to unpickle file {filename}.") from e


def write_parquet(df, filename, pyarrow_schema=None, preserve_index=False):
    """Writes the dataframe to a parquet file using pyarrow with fastparquet as fallback.

    Parameters
    ----------
    df : pandas.DataFrame
        The df to write to parquet.
    filename : str or pathlib.Path
        The path to write to
    pyarrow_schema : pyarrow.Schema, default None
        Datatype schema of parquet file, is inferred 

    Returns
    -------
    
        None

    """
    if _ENGINE == "pyarrow":
        if not pyarrow_schema:
            # TODO make this for loop not slow
            fields = []
            df = df.copy()
            for col in df:
                if col in df.select_dtypes(include="inexact"):
                    fields.append(pa.field(col, pa.float64()))
                elif col in df.select_dtypes(include="integer"):
                    fields.append(pa.field(col, pa.int64()))
                elif col in df.select_dtypes(include=["object", "category"]):
                    df[col] = df[col].astype(str, skipna=True)
                    fields.append(pa.field(col, pa.string()))
                elif col in df.select_dtypes(include="bool"):
                    fields.append(pa.field(col, pa.bool_()))
                elif col in df.select_dtypes(include=["datetime"]):
                    fields.append(pa.field(col, pa.timestamp("ns")))
                elif col in df.select_dtypes(include=["datetimetz"]):
                    fields.append(pa.field(col, pa.timestamp("ns", df[col].dt.tz)))
                else:
                    raise TypeError(f"Column {col} with datatype {df[col].dtype} not supported, enter pyarrow schema")
            pyarrow_schema = pa.schema(fields)
        table = pa.Table.from_pandas(df, preserve_index=preserve_index, schema=pyarrow_schema)
        pq.write_table(table, filename, use_deprecated_int96_timestamps=True)
    elif _ENGINE == "fastparquet":
        fp.write(filename, df, compression="SNAPPY", times="int96", write_index=preserve_index)
    else:
        raise ImportError(
            "Parquet engine not found, install pyarrow or fastparquet and python-snappy to write parquet file."
        )


def dump_data_file(filename, value, to_format=None, preserve_index=True):
    if not to_format:
        import pathlib

        to_format = pathlib.Path(filename).suffix[1:]
        if to_format not in ["parquet", "pkl"]:
            raise ValueError("File format not recognized from file name.")
    if to_format == "parquet":
        #if isinstance(value,pd.Series):
        #    value = value.to_frame()
        if not isinstance(value, pd.DataFrame):
            logger.warn(
                f"We are currently not able to export a {value.__class__} object as {to_format} format. "
                "Exporting with the default configuration."
            )
        try:
            write_parquet(value, filename, preserve_index=preserve_index)
        except:
            logger.warn(f"Exporting as {to_format} format failed. Continuing with the default configuration.")
            to_format = "pkl"

    if to_format == "pkl":
        try:
            with open(filename, "wb") as f:
                pickle.dump(value, f)
        except OverflowError:
            logger.warn("Exporting as pickle was not possible: OverflowError. Retrying using protocol 4...")
            with open(filename, "wb") as f:
                pickle.dump(value, f, protocol=4)


def create_case_key_query(cel_analysis, table, sep=":"):
    """
        Get the columns that make up the primary key and concatenates them
        seperated by sep.

        Parameters
        ----------
        sep: str,optional
            default is ":"
            the separator used for separating the different
            columns, such that a celonis field like:
            BKPF.MANDT|| 'sep' || BKPF.BUKRS|| 'sep' ||...
            is being created

        Returns
        ----------
        _CASE_KEY: str 
            the string making the case_key celonis field 
            e.g. 
            BKPF.MANDT|| 'sep' || BKPF.BUKRS|| 'sep' ||...
        """
    _CASE_KEY = ("|| '" + sep + "' ||").join(
        sorted([table.name + "." + key for key in cel_analysis.datamodel.case_table_key])
    )
    return _CASE_KEY


def push_table(
    cel_datamodel,
    table,
    table_name,
    case_key_col="_CASE_KEY",
    case_key_generated=True,
    add_table_links=True,
    case_key_separator=":",
    reload_datamodel=False,
    if_exists="append",
):
    """
    Pushes the indicated table, with the indicated table name
    back to Celonis.

    Parameters
    ----------
    cel_datamodel : pycelonis.Analysis
        Celonis datamodel where the table will be uploaded.
    table : pandas.DataFrame
        the table to be pushed.
    table_name : str
        name of the table to be pushed.
    case_key_generated : bool
        Whether or not the index of `table` is a concatenation of the case table key.
    add_table_links : bool
        Try to connect to the case table.
    case_key_separator : str
        If `case_key_generated` is True, then specify the separator used to concatenate the primary keys.
    reload_datamodel : bool
        Reload datamodel after pushing table
    if_exists : str, optional
        Default value is "error", if table exists already this will throw an error.
        If "upsert", primary_keys needs to be specified, existing primary keys will be overwritten, new ones appended.
        If "replace", all data will be replaced.
        If "append", new data will be added to the existing data.
    """

    if case_key_generated:
        if table.index.name == case_key_col:
            table = table.reset_index()
        if case_key_col in table.columns:
            new = table["_CASE_KEY"].str.split(case_key_separator, expand=True)
            # new data frame with split value columns
            for i in range(len(cel_datamodel.case_table_key)):
                table[cel_datamodel.case_table_key[i]] = new[i]
            table.drop(["_CASE_KEY"], axis=1, inplace=True, errors="ignore")

    cel_datamodel.push_table(
        table,
        table_name,
        reload_datamodel=reload_datamodel,
        if_exists=if_exists,
        wait_for_finish=(not reload_datamodel),
    )

    if add_table_links:
        cel_datamodel.create_foreign_key(
            table_name, cel_datamodel.process_configurations[0].case_table, cel_datamodel.case_table_key
        )
