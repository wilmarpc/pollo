import argparse
import calendar
import configparser
import csv
import datetime as dt
import gc
import json
import logging
import os
import re
import sys
import time
import warnings
from datetime import time as clock_time
from itertools import repeat
from operator import itemgetter
from pathlib import Path

import holidays as holidays
import numpy as np
import pandas as pd
import timestring
from business_duration import businessDuration
from scipy import sparse

from celonis_ml.data_preprocessing.data_utils import (
    balance_dataset,
    create_case_key_query,
    create_frame_structure,
    dump_data_file,
    impute_numeric_columns,
    load_data_file,
    scale_numeric_columns,
)
from celonis_ml.helpers import coerce_series_to_datetime, create_path, delete_path, get_data_dir, get_example_dir
from pycelonis.pql import PQL, PQLColumn, PQLFilter

warnings.filterwarnings("ignore")


class DataLoader:

    """
    DataLoader manages data, from the extraction from Celonis, to its
    preprocessing to the composition of the dataset. This class supports
    generation of datasets for training and prediction purposes.

    Attributes
    ----------
    cel_analysis: pycelonis.BaseAnalysis
            Celonis Analysis object to exchange data.
    mode : {'development', 'production'}
        run this module in 'development' or 'production'
        mode. This will affect the verbosity and size of
        the data. Not fully implemented yet.
    prediction_settings : str
        settings extracted from the config file
        corresponding to the prediction process.
    purpose : {'training', 'prediction'}
        purpose of the dataset that will be generated. This
        can be either 'training' or 'prediction'. If none
        is specified, will assume 'prediction' as default.
    training_settings : str
        settings extracted from the config file
        corresponding to the prediction process.
    verbosity : {'FATAL', 'ERROR', 'WARN', 'INFO', 'DEBUG'}
        print status updates as the data is processed.
        Equivalent to stating a log level.

    """

    def __init__(
        self,
        cel_analysis,
        shared_selection_url=None,
        purpose="training",
        mode="development",
        sample_size=None,
        balance_dataset=False,
        data_dir=get_data_dir(),
        predefined_use_case=None,
        celonis_filter=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        cel_analysis: pycelonis.BaseAnalysis
            Celonis Analysis object to exchange data.
        purpose : {'training', 'prediction'}
            purpose of the dataset that will be generated. This
            can be either 'training' or 'prediction'. If none
            is specified, will assume 'prediction' as default.
        mode : {'development', 'production'}
            run this module in 'development' or 'production'
            mode. This will affect the verbosity and size of
            the data. Not fully implemented yet.
        sample_size : int
            amount of cases to pull from the case table. Using this parameter is highly advised against, unless you know
            well what you're doing. The way it is implemented right now, pulls the first N cases from the case table, 
            which might as well be unusable for ML, or might belong to the same class (in the case of classification), 
            etc, which would cause problems down the line, use with discretion and if encountering any of the 
            aforementioned errors, try again with a larger N, or disable it entirelly.
        balance_dataset: bool
            define whether or not classes in a data set should be balanced.
        data_dir : path_like
            base folder where to store the data.
        predefined_use_case: celonis_ml.UseCase
            UseCase object to be used as a basis for the filters to be applied.
            If None, the signature will be `generic_dataloader`.
        celonis_filter: list
            Custom PQL filter to be applied when pulling data from Celonis.
        """

        self.purpose = purpose
        self.mode = mode
        self._sample_size = sample_size

        self.celonis_filter = celonis_filter
        self._ct_encoded = None
        self._logger = logging.getLogger(__name__)

        self.use_case = predefined_use_case
        if self.use_case is not None:
            self.signature = self.use_case.signature
        else:
            self.signature = "generic_dataloader"

        # create connection object to celonis
        self.cel_analysis = cel_analysis
        if shared_selection_url:
            self._variables_filters_query = self.cel_analysis.process_shared_selection_url(shared_selection_url)
        else:
            self._variables_filters_query = PQL(self.cel_analysis.published.calculate_variables())

        self._activity_table_name = self.cel_analysis.datamodel.process_configurations[0].activity_table.name
        self._case_table_name = f'"{self.cel_analysis.datamodel.process_configurations[0].case_table.name}"'

        # activity name column
        self._activity_col = self.cel_analysis.datamodel.process_configurations[0].activity_column
        self._activity_table_col = "{}.{}".format(
            self._activity_table_name, self.cel_analysis.datamodel.process_configurations[0].activity_column
        )

        # activity eventtime column
        self._eventtime_col = self.cel_analysis.datamodel.process_configurations[0].timestamp_column
        self._eventtime_table_col = "{}.{}".format(self._activity_table_name, self._eventtime_col)

        # creation date column for the cases
        self.creation_date = f"PU_FIRST({self._case_table_name},{self._eventtime_table_col})"

        # name of case table primary key column
        self._case_key_separator = ":"
        self._case_table_key = create_case_key_query(
            self.cel_analysis, self.cel_analysis.datamodel.process_configurations[0].case_table, self._case_key_separator
        )
        self.set_variables_from_celonis()

        self._logger.info(
            "Connected to Datamodel: {0} and Analysis: {1}".format(
                self.cel_analysis.datamodel.name, self.cel_analysis.name
            )
        )

        self._data_dir = Path(data_dir)
        self.training_settings = {}
        self.training_settings_file = os.path.join(self._data_dir, "training", "training_settings.pkl")

        if self.purpose == "prediction":
            assert os.path.isdir(
                os.path.join(self._data_dir, "training")
            ), "You must train before predicting. Trained model not found."
            self.training_settings = load_data_file(self.training_settings_file)

        # store all necessary side information in this dictionary
        if self.purpose == "training":
            if os.path.isfile(self.training_settings_file):
                self.training_settings = load_data_file(self.training_settings_file)

            if sample_size:
                self.training_settings["sample_size"] = sample_size
            self._SEED = 123

        if self.cel_analysis.datamodel.case_table_key and len(self.cel_analysis.datamodel.case_table_key) > 1:
            self.training_settings["case_key_generated"] = True

        # Store parameters for future runs
        execution_fingerprint = (
            (str(self.mode) + str(self._sample_size) + str(self._data_dir)) + self.purpose + self.signature
        )

        self._logger.info(f"Checking fingerprint... {execution_fingerprint}")
        import pickle

        # execution_fingerprint += str(pickle.dumps(config))
        fingerprint_file = os.path.join(self._data_dir, self.purpose, f"{self.purpose}_fingerprint.pkl")

        if os.path.isfile(fingerprint_file):
            self._logger.info("Fingerprint found")
            previous_fingerprint = load_data_file(fingerprint_file)
            if previous_fingerprint != execution_fingerprint:
                delete_path(os.path.join(self._data_dir, self.purpose))
                self._logger.info(
                    "Fingerprints did not match."
                    + " Thus the data folder is deleted and "
                    + "data will be gathered from Celonis"
                )
        else:
            self._logger.info("Fingerprint not found")
            delete_path(self._data_dir / self.purpose)
            if self.purpose == "training":
                self.training_settings = {}
            elif self.purpose == "prediction":
                self.training_settings = load_data_file(self.training_settings_file)
            if sample_size:
                self.training_settings["sample_size"] = sample_size
            self._SEED = 123

        # create directories
        create_path(self._data_dir)
        create_path(os.path.join(self._data_dir, "prediction"))
        create_path(os.path.join(self._data_dir, "training"))
        dump_data_file(fingerprint_file, execution_fingerprint)

        # Don't show warnings
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        # PQL QUERY STATEMENTS FOR PULLING FROM CELONIS ON CASE LEVEL
        self._INDEX_QUERY = PQL(self._variables_filters_query)
        self._CONTINUOUS_QUERY = PQL(self._variables_filters_query)
        self._CATEGORICAL_QUERY = PQL(self._variables_filters_query)
        self._USE_CASE_QUERY = PQL(self._variables_filters_query)

    @property
    def purpose_filter(self):
        try:
            self._purpose_filter_s
        except:
            self._purpose_filter_s = self.training_settings.get(f"{self.signature}_filter_string")
        if self._purpose_filter_s is None:
            return None
        elif self.purpose == "training":
            cel_filter = [f"{self._purpose_filter_s} = 'train' "]
        elif self.purpose == "prediction":
            cel_filter = [f"{self._purpose_filter_s} = 'predict' "]
            if hasattr(self, "reaction_time") and self.reaction_time is not None:
                cel_filter += [
                    f"ROUND(DAYS_BETWEEN(TO_TIMESTAMP('{self.today_help}', 'YYYY-MM-DD '),"
                    f"{self.due_date})) IN ({str(self.reaction_time)[1:-1]})"
                ]
            elif hasattr(self, "event_count") and self.event_count is not None:
                cel_filter += [
                    " PU_MAX({0},{1},PROCESS_ORDER({1}))) = {2} ".format(
                        self._case_table_name, self._activity_table_col, self.event_count
                    )
                ]
        else:
            cel_filter = [f"{self._purpose_filter_s} != 'ignore' "]
        return [PQLFilter(f) for f in cel_filter]

    @property
    def case_query(self):
        if self.purpose == "prediction":
            data_to_pull = self.training_settings.get(f"case_query_{self.signature}")
        else:
            data_to_pull = self._INDEX_QUERY + self._CONTINUOUS_QUERY + self._CATEGORICAL_QUERY + self._USE_CASE_QUERY
        return data_to_pull

    @property
    def _y_train_file(self):
        return os.path.join(self._data_dir, self.purpose, "y_train.parquet")

    @property
    def dataset_file(self):
        return os.path.join(self._data_dir, self.purpose, (f"dataset_{self.act_id}_{self.signature}.parquet"))

    @property
    def _cases_raw_file(self):
        return os.path.join(self._data_dir, self.purpose, f"cases_raw_{self.signature}.parquet")

    @property
    def _activities_raw_file(self):
        return os.path.join(self._data_dir, self.purpose, f"activities_raw_{self.signature}.parquet")

    # path to final case frame
    @property
    def _cases_final_file(self):
        return os.path.join(self._data_dir, self.purpose, f"cases_final_{self.signature}.parquet")

    @property
    def _activities_final_file(self):
        return os.path.join(self._data_dir, self.purpose, f"activities_final_{self.signature}.parquet")

    def set_variables_from_celonis(self):
        q = self._variables_filters_query.variables

        self.due_date = q.get("due_date")
        self.today_help = q.get("today_help")
        self.target_variable_activity = re.sub(r"^'|'$", "", q.get("real_date")) if q.get("real_date") else None
        self.buffer = q.get("buffer")
        self.data_ext_start_date = timestring.Date(q.get("data_ext_start_date")).date.strftime("%Y-%m-%d")
        self.case_features = q.get("case_features")

    def create_case_key_query(self, table, sep=":"):
        """
        Get the columns that make up the primary key and concatenates them
        seperated by sep.

        Parameters
        ----------
        sep: str,optional
            default is ":"
            the seperator used for separating the different
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
        case_table_key = self.cel_analysis.datamodel.case_table_key
        case_table_key = [case_table_key] if not isinstance(case_table_key, list) else case_table_key
        _CASE_KEY = ("|| '" + sep + "' ||").join(
            sorted([f'"{table.name}"."{key}"' for key in case_table_key])
        )
        self._logger.info("The _CASE_KEY is " + _CASE_KEY + " for table: " + table.name)
        return _CASE_KEY

    def get_use_case_query(self, sheet, table):
        """
        Checks if TARGET_VARIABLE and CASE_STATE are in the input table. Additionally, gets the PQL queries used to
        generate these columns.

        Parameters
        ----------
        sheet : str
            Name of the sheet where to find the input table for python.
        table : str
            Name of the input table.

        Returns
        -------
            TARGET_VARIABLE_QUERY

            CASE_STATE_QUERY
        """
        pql = self.cel_analysis.published.sheets.names[sheet].components.names[table].pql_query

        TARGET_VARIABLE_QUERY = PQL(self._variables_filters_query)
        CASE_STATE_QUERY = PQL(self._variables_filters_query)

        for col in pql.columns:
            if "target_variable" == col.name.lower():
                TARGET_VARIABLE_QUERY += col
            elif "case_state" == col.name.lower():
                CASE_STATE_QUERY += col
                self._purpose_filter_s = col.query
                self.training_settings[f"{self.signature}_filter_string"] = col.query

        assert (
            len(TARGET_VARIABLE_QUERY.columns) > 0
        ), f"Column TARGET_VARIABLE not found in {table}\
            please specify here the Class or Continuous variable you want the model to predict"

        assert (
            len(CASE_STATE_QUERY.columns) > 0
        ), f"Column CASE_STATE not found in {table}\
            please specify here for each case if it is 'open' or 'closed' or 'missing data' for this use case."

        return TARGET_VARIABLE_QUERY + CASE_STATE_QUERY

    def get_columns_from_feature_recommendation(self):
        """
        Reads in the variables selected in the feature recommendation sheet
        and adds them to the input features.
        """

        if not hasattr(self, "case_features") or len(self.case_features) < 3:
            return []
        q = PQL(self._variables_filters_query)
        for c in [col.replace("'", "").strip() for col in self.case_features.split(",")]:
            q += PQLColumn(c, c)
        return q

    def _get_case_feat_query(self, sheet, table):
        """
        Pulls the top 100 rows from case table, checks for categorical values
        and matches it with the datatype and matches each field to its type,
        i.e. is it an index, a categorical feature, the label or continuous
        Updates the queries and adds the filters from the table to the
        `_case_table_filter`

        Parameters
        ----------
        sheet_name : str
            Name of the sheet where to find the input table for
            python.
        table_to_pull : str
            Name of the table tp be pulled from Celonis.


        Returns
        -------
        list of celonis_field
            list of celonis fields that can be used as a query
            in pull_data.

        """
        pql = self.cel_analysis.published.sheets.names[sheet].components.names[table].pql_query

        pql += self.get_columns_from_feature_recommendation()
        pql += self._variables_filters_query
        pql.limit = 10
        table = self.cel_analysis.get_data_frame(pql)  # , variables=self.cel_analysis.published.calculate_variables())

        # Checking the inputs for types etc.
        INDEX_QUERY = PQL(pql.filters) + PQL(self._variables_filters_query)
        CATEGORICAL_QUERY = PQL(self._variables_filters_query)
        CONTINUOUS_QUERY = PQL(self._variables_filters_query)

        for col, i in zip(pql.columns, range(table.shape[1])):
            if col.name.lower() in [
                "target_variable",
                "case_state",
                *[l.lower() for l in self.cel_analysis.datamodel.case_table_key],
            ]:
                continue
            elif "object" in table.dtypes[i].name:
                if "object" not in table.dtypes[i].name:
                    col.query += """||''"""
                CATEGORICAL_QUERY += col
            elif ("int" in table.dtypes[i].name) or ("float" in table.dtypes[i].name):
                CONTINUOUS_QUERY += col

        INDEX_QUERY += PQLColumn(self._case_table_key, "_CASE_KEY")

        return INDEX_QUERY, CATEGORICAL_QUERY, CONTINUOUS_QUERY

    def _extract_top_categories(self, categorical_query, max_n_cat=None, quantile=None):
        """
        Takes the categorical variables that we want to query for and takes the
        maximum of the top 80% quantile or the top max_n_rows rows, all other
        categories are matched to 'other' NAs are matched to 'missing'.

        Parameters
        ----------
        max_n_rows : int
            the maximum amount of category_values allowd per
            category.
        quantile : float
            the quantile of the data selected by label rate and
            case count that we want to include as important
            category.

        Returns
        -------
        list of PQLColumn
            list of PQL columns that can be used as a query
            in pull_data.
        """

        # TODO: make sure we pull the important features, ie not only look for
        # netvalue and case count, but also take the label field into account
        if self.purpose == "training":
            cat_clean = PQL()
            for col in categorical_query.columns:
                cat_clean.add(self._cluster_category(col))

            # TODO: this can be erased i think
            # Save dictionary of categories to pickle file
            self.training_settings["categorical_query"] = cat_clean

        # Loading the pickle file for predictions
        if self.purpose == "prediction":
            cat_clean = self.training_settings.get("categorical_query")

        return cat_clean

    def one_hot_encode_table(self, sheet_name, table_name):
        """
        Gets the case table and preprocesses it making it ready for ML models.

        Returns
        -------
        pandas.DataFrame
            the preprocessed, one hot encoded case table ready
            to be used for training/predicting with ML models.

        """
        # TODO: add celonis filter
        # purpose is training here the query is being created, if purpose is prediction the query is loaded
        if self.purpose == "training" and os.path.isfile(self._cases_final_file):
            ct_encoded = load_data_file(self._cases_final_file, from_format="parquet")
        else:
            if self.purpose == "training":
                self._INDEX_QUERY, self._CATEGORICAL_QUERY, self._CONTINUOUS_QUERY = self._get_case_feat_query(
                    sheet_name, table_name
                )
                # one hote encoding of categorical query
                self._CATEGORICAL_QUERY += self._extract_top_categories(self._CATEGORICAL_QUERY)
                # extracting the target variable and case_state_query
                self._USE_CASE_QUERY = self.get_use_case_query(sheet_name, table_name)
                # save the case query
                self.training_settings[f"case_query_{self.signature}"] = self.case_query

            # pull all the queries above to create the df
            cases = self.pull_data(self.case_query, self.celonis_filter, limit=self._sample_size)
            assert cases.shape[0] > 0 and len(set(cases.columns.values)) == cases.shape[1], (
                "Case table is either " "empty or has duplicate columns."
            )

            # drop columns with only nan values
            if self.purpose == "training":
                self.training_settings["nan_col_names"] = cases.columns[cases.isna().all()]

            cases.drop(self.training_settings["nan_col_names"], axis=1, inplace=True)

            ct_encoded = self._preprocess_cases_table(cases)

            assert len(set(cases.columns.values)) == cases.shape[1]
            assert len(set(ct_encoded.columns.values)) == ct_encoded.shape[1]
        return ct_encoded

    def _preprocess_cases_table(self, cases):
        """
        takes the python_input_table and one hot encodes the categorical field,
        labelling very rare categories as other, cleaning the dtypes and
        selecting the wanted cases for training and prediction.

        Parameters
        ----------
        cases : pandas.DataFrame
            dataframe one case level that is to be cleaned.
        table_name : str
            if 'ct' then this is for the case table, if 'at' to
            be added.

        Returns
        -------
        cases : pandas.DataFrame
            the input table.
        feature_matrix : pandas.DataFrame
            the preprocessed, one hot encoded case table ready
            to be used for training/predicting with ML models.

        """

        # fill missing categorical fields with 'missing'
        cat_col_names = [
            col.name
            for col in self._CATEGORICAL_QUERY.columns
            if col.name not in self.training_settings["nan_col_names"]
        ]
        cases[cat_col_names] = cases[cat_col_names].fillna("missing")

        # one hot encoding of pre binned categories
        cases.set_index("_CASE_KEY", inplace=True)

        cols_to_encode = [cases.columns.values[i] for i, c in enumerate(cases.dtypes) if c == "object"]
        cols_to_encode = set(cols_to_encode) - set(["TARGET_VARIABLE"])
        ct_encoded = pd.get_dummies(cases, columns=cols_to_encode)

        # get numeric column names
        if self.purpose == "training":
            num_col_names = [
                col.name
                for col in self._CONTINUOUS_QUERY.columns
                if col.name not in self.training_settings["nan_col_names"]
            ]
            self.training_settings["num_col_names"] = num_col_names
        else:
            num_col_names = self.training_settings["num_col_names"]

        ct_encoded[num_col_names], self.training_settings = impute_numeric_columns(
            ct_encoded[num_col_names], self.purpose, self.training_settings, "imputer_cases_table"
        )
        ct_encoded[num_col_names], self.training_settings = scale_numeric_columns(
            ct_encoded[num_col_names], self.purpose, self.training_settings, "scaler_cases_table"
        )
        ct_encoded, self.training_settings = create_frame_structure(
            ct_encoded,
            self.purpose,
            self.training_settings,
            self.signature,
            self._cases_final_file,
            filler=0,
            table_name="case_table",
        )

        self._logger.info(f"Obtaining data for {self.purpose} with cases table of shape: {ct_encoded.shape}")

        return ct_encoded

    def pull_data(self, cel_query, cel_filter=None, apply_filters=True, limit=None, **kwargs):
        """
        Pulls the indicated query from celonis, with the cel_filters
        and the DataLoader filters

        Parameters
        ----------
        cel_query : pandas.DataFrame
            the table to be pushed.
        cel_filter : str
            name of the table to be pushed.

        Returns
        ----------
        df : pandas.DataFrame
            the pull OLAP table in a DataFrame.

        """
        total_query = PQL(cel_query)

        if cel_filter is not None:
            total_query.add(cel_filter)
        # filters for predefined use cases
        if self.use_case is not None:
            total_query.add(self.use_case.pql_filter(purpose=self.purpose))
        if apply_filters:
            # generall filtering for all purposes
            if self.celonis_filter is not None:
                total_query.add(self.celonis_filter)
            # dynamic purpose filtering
            if self.purpose_filter is not None:
                total_query.add(self.purpose_filter)

        # getting the data from celonis
        if limit:
            total_query.limit = limit
        df = self.cel_analysis.get_data_frame(total_query, **kwargs)

        return df

    def _cluster_category(self, pql_column, quant=0.99, max_n_cat=500, cel_filter=None):
        """ Counts how many cases there are for each category in specific column, and then remains only with the
        `max_n_cat` that are in the top `quant` quantile. Then it creates a PQL query out of this.
        """
        DESC_QUERY = PQL(self._variables_filters_query)
        DESC_QUERY += PQLColumn(f"COUNT_TABLE( {self._case_table_name} )", "COUNT")
        DESC_QUERY += pql_column

        # Additional filters could be included, e.g. for reaction_time
        cat_df = self.pull_data(DESC_QUERY, cel_filter)

        cat_df = cat_df[cat_df["COUNT"] > cat_df["COUNT"].quantile(1 - quant)]
        cat_df = cat_df.nlargest(max_n_cat, "COUNT")
        cat_df = cat_df.dropna()
        cat_string = "'" + "','".join(cat_df[pql_column.name].tolist()) + "'"
        new_query = f"CASE WHEN ISNULL({pql_column.query})=1 THEN 'missing'\
                          WHEN {pql_column.query} in ({cat_string}) THEN {pql_column.query} \
                          ELSE 'other' END"

        pql_column.query = new_query
        return pql_column

    def obtain_data(self, cases_only=False, **kwargs):
        """
        This method is the entry point for DataLoader. Gets both case and activity table and preprocesses them making
        them ready for ML models.

        Returns
        -------
        ct_encoded : pandas.DataFrame
            the preprocessed, one hot encoded case table ready
            to be used for training/predicting with ML models.
        at_encoded : pandas.DataFrame
            the preprocessed, one hot encoded activity table
            ready to be used for training/predicting with ML
            models.

        """
        input_cases_sheet = "Model Setup 5/5"
        input_cases_table = "model_input_cases_table"
        reaction_time, event_count = None, None
        for k, v in kwargs.items():
            if k == "input_cases_sheet":
                input_cases_sheet = v
            if k == "input_cases_table":
                input_cases_table = v
            if k == "reaction_time":
                reaction_time = v if isinstance(v, list) else [v]
            if k == "event_count":
                event_count = v

        # Getting reaction time or event_count
        if reaction_time is None:
            if "reaction_time" in self._variables_filters_query.variables.keys():
                reaction_time = [int(r) for r in self._variables_filters_query.variables["reaction_time"].split(",")]
        if event_count is None:
            if "event_count" in self._variables_filters_query.variables.keys():
                event_count = self._variables_filters_query.variables["event_count"]
        # TODO: read in signature
        self.reaction_time = reaction_time
        self.event_count = event_count

        self.act_id = self._set_identifier(cases_only)
        if self.purpose == "training" and os.path.isfile(self.dataset_file):
            all_encoded = load_data_file(self.dataset_file, from_format="parquet")
        else:
            if self._ct_encoded is None or self.purpose == "prediction":
                self._ct_encoded = self.one_hot_encode_table(input_cases_sheet, input_cases_table)

            at_encoded = None
            if not cases_only:
                from celonis_ml.data_preprocessing import ActivityManager

                case_table_filters = PQL(self._variables_filters_query) + self.purpose_filter
                if self._sample_size:
                    # If we want to sample, we pull only the activities for the cases we already pulled.
                    act_key_query = self.create_case_key_query(
                        self.cel_analysis.datamodel.process_configurations[0].activity_table, self._case_key_separator
                    )
                    case_table_filters += PQLFilter(f"{act_key_query} IN ({str(list(self._ct_encoded.index))[1:-1]})")

                am = ActivityManager(
                    self.cel_analysis,
                    self._data_dir,
                    self.purpose,
                    self.training_settings,
                    self.training_settings_file,
                    self._activities_final_file,
                    self.signature,
                    self.target_variable_activity,
                    case_table_filters,
                )

                at_encoded = am.pull_activity_table(reaction_time=reaction_time, due_date=self.due_date)
                self.training_settings.update(am.training_settings)

            if at_encoded is not None:
                all_encoded = self._ct_encoded.merge(at_encoded, how="inner", left_index=True, right_index=True)
            else:
                all_encoded = self._ct_encoded

            all_encoded = self.add_date_dep_features(
                input_case_table=all_encoded, date_col=self.creation_date, date_name="creation_date"
            )
            if self.due_date is not None:
                all_encoded = self.add_date_dep_features(
                    input_case_table=all_encoded, date_col=self.due_date, date_name="due_date"
                )

            assert len(set(all_encoded.columns.values)) == all_encoded.shape[1], "there are duplicate columns."
            assert all_encoded.shape[0] > 0, "final dataset is empty."
            all_encoded, self.training_settings = create_frame_structure(
                all_encoded,
                self.purpose,
                self.training_settings,
                self.signature,
                self.dataset_file,
                filler=0,
                table_name=f"X_train_{self.act_id}",
            )

        if self.purpose == "training":
            dump_data_file(self.training_settings_file, self.training_settings)

        if "TARGET_VARIABLE" in all_encoded.columns:
            y = all_encoded["TARGET_VARIABLE"]
            all_encoded = all_encoded.drop(["TARGET_VARIABLE", "CASE_STATE"], axis=1, errors="ignore")
            self._logger.info(f"Final dataset for {self.purpose} has shape: {all_encoded.shape}")

            return all_encoded, y
        else:
            self._logger.info(f"Final dataset for {self.purpose} has shape: {all_encoded.shape}")
            return all_encoded

    def _set_identifier(self, cases_only):
        """
        Searches the attributes to give back the corresponding reaction time or event count to save/load correct df
        """
        if cases_only:
            s = "no_activities"
        else:
            if self.reaction_time is not None:
                s = f"react_{self.reaction_time}"
            elif self.event_count is not None:
                s = f"event_count_{self.event_count}"
            else:
                s = "generic_activities"
        return s

    def add_date_dep_features(self, input_case_table=None, date_col=None, date_name="creation_date"):
        """
        Creates features based on the days before the due date
        like amount of items, delay rate, open items in process,

        Parameters
        ----------
        case_table : 

        Returns
        -------
        pandas.DataFrame
            ordered by day and containing process
            information of that day

        """
        if date_col is None:
            date_col = self.creation_date
        # query for celonis
        q = PQL(self._variables_filters_query)
        q += PQLColumn("ROUND_DAY({})".format(date_col), date_name)
        q += PQLColumn("COUNT_TABLE({})".format(self._case_table_name), "Number_of_cases_on_the_same_day")
        q += PQLColumn("DAY_OF_WEEK({})".format(date_col), "WEEKDAY")

        cel_filter = None
        if date_name == "due_date" and hasattr(self, "reaction_time") and self.reaction_time is not None:
            cel_filter = PQLFilter(f"{date_col} < ADD_DAYS({self.due_date}, {max(self.reaction_time)})")

        df = self.pull_data(q, cel_filter, apply_filters=False)
        df["days_until_end_of_month"] = df[date_name].dt.daysinmonth - df[date_name].dt.day
        df["_month"] = df[date_name].dt.month
        df[date_name] = df[date_name].dt.strftime("%Y-%m-%d")
        df = df.set_index(date_name)
        df_w = pd.get_dummies(df["WEEKDAY"], prefix="Weekday_is:")
        df = df.drop(["WEEKDAY"], axis=1)
        df = df.merge(df_w, how="inner", left_index=True, right_index=True)
        df.columns += f"_{date_name}"

        # if used as addition features, brought to case level and then joined via help df
        if input_case_table is not None:
            q = PQL(self._variables_filters_query)
            q += PQLColumn("ROUND_DAY({})".format(date_col), date_name)
            q += PQLColumn(self._case_table_key, "_CASE_KEY")
            df_help = self.pull_data(q)
            df_help = df_help.set_index("_CASE_KEY")
            df_help["date_text"] = df_help[date_name].dt.strftime("%Y-%m-%d")
            df_help = df_help.merge(df, how="left", left_on="date_text", right_index=True)
            df_help = df_help.drop(["date_text", date_name], axis=1)
            input_case_table = input_case_table.merge(df_help, how="left", left_index=True, right_index=True)
            if self.purpose == "training":
                if not self.training_settings.get(f"add_date_dep_features_{self.signature}"):
                    self.training_settings[f"add_date_dep_features_{self.signature}"] = {"date_name": date_col}
                else:
                    self.training_settings[f"add_date_dep_features_{self.signature}"].update({"date_name": date_col})
            # TODO: how to integrate this with create framestructure
            # input_case_table = self._create_frame_structure(input_case_table)
            return input_case_table
        else:
            return df

    def working_hours_between(self, df, col1, col2, bday_start=5, bday_end=22, country="DE"):
        # Business open hour
        biz_open_time = dt.time(bday_start, 0, 0)

        # Business close time
        biz_close_time = dt.time(bday_end, 0, 0)

        # Weekend list. 5-Sat, 6-Sun
        weekend_list = [5, 6]

        # holidays
        if country == "DE":
            holiday_list = holidays.DE()
        elif country == "US":
            holiday_list = holidays.US()
        else:
            holiday_list
        # Business duration 'day','hour','min','sec'
        unit_hour = "hour"

        # Applying the function to entire dataframe
        df["work_hours_between_{0}_{1}".format(col1, col2)] = list(
            map(
                businessDuration,
                df[col1],
                df[col2],
                repeat(biz_open_time),
                repeat(biz_close_time),
                repeat(weekend_list),
                repeat(holiday_list),
                repeat(unit_hour),
            )
        )
        return df[["work_hours_between_{0}_{1}".format(col1, col2)]]

