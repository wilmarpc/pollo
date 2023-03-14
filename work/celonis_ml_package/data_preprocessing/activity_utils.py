import numpy as np
import pandas as pd

from celonis_ml.data_preprocessing.data_utils import (
    create_case_key_query,
    create_frame_structure,
    dump_data_file,
    impute_numeric_columns,
    scale_numeric_columns,
)
from pycelonis.pql import PQL, PQLColumn, PQLFilter


def get_common_activities_by_activation_count(
    cel_analysis, variables_filters_query, target_variable_activity, quantile=0.8, max_n_rows=500, **kwargs
):
    """
    Gets common activites under the criteria: activation count
    # taking only the top quant quantile of the data as distinct categories
    # taking only the top max_n_rows of the data as distinct categories
    # list of distinct categories for celonis
    """
    activity_column = "{}.{}".format(
        cel_analysis.datamodel.process_configurations[0].activity_table.name,
        cel_analysis.datamodel.process_configurations[0].activity_column,
    )

    q = PQL(variables_filters_query)
    q += [
        PQLColumn(f"DISTINCT {activity_column} || '__' || ACTIVATION_COUNT({activity_column})", "ACT_AND_ACT_COUNT"),
        PQLColumn(f"ACTIVATION_COUNT({activity_column})", "ACT_COUNT"),
        PQLColumn(f"COUNT_TABLE( {cel_analysis.datamodel.process_configurations[0].case_table.name} )", "COUNT"),
        PQLColumn(f"{activity_column}", "ACT"),
    ]

    act_df = cel_analysis.get_data_frame(q, **kwargs)

    act_df = act_df[act_df["COUNT"] > act_df["COUNT"].quantile(1 - quantile)]
    act_df = act_df.nlargest(max_n_rows, "COUNT")
    act_df = act_df.dropna()
    # TODO: REMOVE THIS VIA PQL FILTER
    act_df = act_df[act_df.ACT != target_variable_activity]
    act_string = str(act_df["ACT_AND_ACT_COUNT"].tolist())[1:-1]
    acts_unique = act_df["ACT"].unique()

    return act_string, acts_unique


def pull_common_activities(cel_analysis, variables_filters_query, due_date, separator=":", **kwargs):
    """
    Pulls the activity table under the criteria: activation count
    # Activity name
    # Time to due date for each activity and the activation count
    # Process Order of each activity 
    """
    activity_column = "{}.{}".format(
        cel_analysis.datamodel.process_configurations[0].activity_table.name,
        cel_analysis.datamodel.process_configurations[0].activity_column,
    )
    eventtime_table_col = "{}.{}".format(
        cel_analysis.datamodel.process_configurations[0].activity_table.name,
        cel_analysis.datamodel.process_configurations[0].timestamp_column,
    )

    q = PQL(variables_filters_query)
    q += PQLColumn(
        create_case_key_query(cel_analysis, cel_analysis.datamodel.process_configurations[0].case_table, separator),
        "_CASE_KEY",
    )

    q += PQLColumn(activity_column, "name")
    q += PQLColumn("DAYS_BETWEEN({0},{1})".format(eventtime_table_col, due_date), "TIME_TO_DUE_DATE")
    q += PQLColumn("PROCESS_ORDER({0})||''".format(activity_column), "PROCESS_ORDER")
    q += PQLColumn("{0}".format(eventtime_table_col), "EVENTTIME")

    act_df = cel_analysis.get_data_frame(q, **kwargs)

    return act_df


def pull_common_activities_by_activation_count(
    cel_analysis, act_string, variables_filters_query, due_date, separator=":", **kwargs
):
    """
    Pulls the activity table under the criteria: activation count
    # Activity name
    # Time to due date for each activity and the activation count
    # Process Order of each activity 
    """
    activity_column = "{}.{}".format(
        cel_analysis.datamodel.process_configurations[0].activity_table.name,
        cel_analysis.datamodel.process_configurations[0].activity_column,
    )
    eventtime_table_col = "{}.{}".format(
        cel_analysis.datamodel.process_configurations[0].activity_table.name,
        cel_analysis.datamodel.process_configurations[0].timestamp_column,
    )

    q = PQL(variables_filters_query)
    q += PQLColumn(
        create_case_key_query(cel_analysis, cel_analysis.datamodel.process_configurations[0].case_table, separator),
        "_CASE_KEY",
    )

    print("Act_string:", act_string)

    q += PQLColumn(
        "CASE WHEN {0}||'__'||ACTIVATION_COUNT({0}) in({1}) THEN {0}||'_Occurence_'||ACTIVATION_COUNT({0}) "
        "ELSE PROCESS_ORDER({0})||'_Activity is_'||'other' END".format(activity_column, act_string),
        "name",
    )
    q += PQLColumn("DAYS_BETWEEN({0},{1})".format(eventtime_table_col, due_date), "TIME_TO_DUE_DATE")
    q += PQLColumn("PROCESS_ORDER({0})||''".format(activity_column), "PROCESS_ORDER")
    q += PQLColumn("{0}".format(eventtime_table_col), "EVENTTIME")

    act_df = cel_analysis.get_data_frame(q, **kwargs)

    return act_df


def get_number_times_activites_performed(
    cel_analysis, variables_filters_query, acts_unique, due_date, reaction_time, separator=":", **kwargs
):
    """
    Calculates the activation count of each activity before reaction time
    """

    case_key_query = create_case_key_query(
        cel_analysis, cel_analysis.datamodel.process_configurations[0].case_table, separator
    )
    case_table_name = f'"{cel_analysis.datamodel.process_configurations[0].case_table.name}"'
    activity_table_col = f"{cel_analysis.datamodel.process_configurations[0].activity_table.name}.{cel_analysis.datamodel.process_configurations[0].activity_column}"
    eventtime_table_col = f"{cel_analysis.datamodel.process_configurations[0].activity_table.name}.{cel_analysis.datamodel.process_configurations[0].timestamp_column}"

    q = PQL(variables_filters_query)
    q += PQLColumn(case_key_query, "_CASE_KEY")

    for act in acts_unique:
        q += PQLColumn(
            f"PU_COUNT({case_table_name}, {activity_table_col}, {activity_table_col} = '{act}' "
            f"AND DAYS_BETWEEN({eventtime_table_col}, {due_date}) > {min(reaction_time)})",
            f"Number of Times {act} has been performed",
        )

    df_count = cel_analysis.get_data_frame(q, **kwargs)
    return df_count.set_index("_CASE_KEY")


def one_hot_encode_activities_by_activation_count(
    at, at_count, purpose, training_settings, training_settings_file, activities_final_file, signature
):
    """
        Takes the activity table with the activities together with the
        activation count and only returns information on up to three
        activations counts and gives the remaining time.

        Returns
        -------
        pandas.DataFrame
            the processed activity table
            ready to be used for training/predicting with ML
            models.

        """

    at_processed = at.pivot(index="_CASE_KEY", columns="name", values="TIME_TO_DUE_DATE")
    at_processed.columns = "Time_until_due_date_from:_" + at_processed.columns

    at_help = at.pivot(index="_CASE_KEY", columns="name", values="PROCESS_ORDER")
    at_help = at_help.astype(np.float64)
    at_help.columns = "Process Order of: " + at_help.columns

    at_processed = at_processed.merge(at_help, how="left", left_index=True, right_index=True)

    at["EVENTTIME"] = pd.to_datetime(at["EVENTTIME"])
    at_gap = at[["_CASE_KEY", "PROCESS_ORDER", "EVENTTIME"]]

    at_gap = at_gap.sort_values(by=["_CASE_KEY", "EVENTTIME"])

    # Group by taking long time in prediction
    at_gap["GAP"] = at_gap.groupby("_CASE_KEY")["EVENTTIME"].diff().dt.total_seconds()
    at_gap = at_gap.pivot(index="_CASE_KEY", columns="PROCESS_ORDER", values="GAP")
    at_gap = at_gap.drop(columns=["1"])

    at_gap.columns = "Time Until Process Order: " + at_gap.columns

    at_processed = at_processed.merge(at_gap, how="left", left_index=True, right_index=True)
    at_processed = at_count.merge(at_processed, how="left", left_index=True, right_index=True)
    # create frame structure
    at_processed, training_settings = create_frame_structure(
        at_processed,
        purpose,
        training_settings,
        signature,
        activities_final_file,
        filler=0,
        table_name="activity_table",
    )

    # fill nans with median
    at_tmp, training_settings = impute_numeric_columns(
        at_processed, purpose, training_settings, "imputer_act_table_activation_count"
    )
    at_processed = pd.DataFrame(at_tmp, columns=at_processed.columns, index=at_processed.index)

    # Apply scaler
    at_tmp, training_settings = scale_numeric_columns(
        at_processed, purpose, training_settings, "scaler_act_table_activation_count"
    )
    at_processed = pd.DataFrame(at_tmp, columns=at_processed.columns, index=at_processed.index)

    dump_data_file(training_settings_file, training_settings)
    return at_processed, training_settings


def one_hot_encode_activities_by_event_count(cel_analysis, event_count, selection_quantile=0.95, **kwargs):
    """
    WARNING!!! THIS FEATURE HAS NOT YET BEEN RELEASED. THIS CODE IS NOT YET TO BE TESTED.
    Takes the activity table with the activities together with the step
    in the process and some more information and one hot encodes the
    whole process flow, creating many columns, labelling very rare act
    step combinations as other,cleaning the dtypes and selecting the
    wanted cases for training and prediction.

    Returns
    -------
    pandas.DataFrame
        the one hot encoded activity table
        ready to be used for training/predicting with ML
        models.

    """
    case_table_name = f'"{cel_analysis.datamodel.process_configurations[0].case_table.name}"'
    activity_table_col = f"{cel_analysis.datamodel.process_configurations[0].activity_table.name}.{cel_analysis.datamodel.process_configurations[0].activity_column}"
    eventtime_table_col = (
        f"{cel_analysis.datamodel.process_configurations[0].activity_table.name}.{cel_analysis.datamodel.timestamp_column}"
    )

    max_events = PQL(
        PQLColumn(
            f"QUANTILE(PU_MAX({case_table_name}, PROCESS_ORDER({activity_table_col})), {selection_quantile})", "max"
        )
    )

    max_events = cel_analysis.get_data_frame(max_events, **kwargs).iloc[0, 0]

    max_events_filter = PQLFilter(f"PU_MAX({case_table_name},PROCESS_ORDER({eventtime_table_col})) < {max_events}")

    activities = 1  # _pull_activity_table()
    at_encoded = 1  # _preprocess_activities_table(activities)
    at_encoded = 1  # _create_frame_structure(at_encoded, table_name='activity_table')

    return at_encoded, max_events_filter, activities
