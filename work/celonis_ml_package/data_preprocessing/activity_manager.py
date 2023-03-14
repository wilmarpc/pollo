import logging

from pycelonis.pql import PQL, PQLColumn, PQLFilter
from celonis_ml.data_preprocessing.activity_utils import (get_common_activities_by_activation_count, 
                                                     pull_common_activities_by_activation_count,
                                                     get_number_times_activites_performed,
                                                     one_hot_encode_activities_by_activation_count,
                                                     one_hot_encode_activities_by_event_count,
                                                     pull_common_activities)

class ActivityManager():
    def __init__(self, 
                 cel_analysis,
                 data_dir,
                 purpose,
                 training_settings,
                 training_settings_file,
                 activities_final_file,
                 signature,
                 target_variable_activity,
                 case_table_filters=None,
                 extra_filters=None):


        self.cel_analysis = cel_analysis
        self.data_dir = data_dir
        self.purpose = purpose
        self.training_settings = training_settings
        self.training_settings_file = training_settings_file
        self.activities_final_file = activities_final_file
        self.signature = signature
        self.target_variable_activity = target_variable_activity
        self.case_table_filters = case_table_filters
        self.extra_filters = extra_filters

        self._logger = logging.getLogger(__name__)


    def pull_activity_table(self,
                            reaction_time=None,
                            event_count=None,
                            due_date=None,
                            quantile=0.8,
                            max_n_rows=500,
                            **kwargs):

        reaction_time = [int(n) for n in reaction_time.split(",")] if isinstance(reaction_time, str) else reaction_time
        extract_filter = PQL(self.case_table_filters)
        eventtime_table_col = '{}.{}'.format(self.cel_analysis.datamodel.process_configurations[0].activity_table.name,
                                             self.cel_analysis.datamodel.process_configurations[0].timestamp_column)

        if reaction_time:
            if self.purpose == "training":
                extract_filter += PQLFilter(f"DAYS_BETWEEN({eventtime_table_col}, {due_date}) >"\
                                            f" {min(reaction_time)}")
            
            act_string, act_unique = get_common_activities_by_activation_count(self.cel_analysis, extract_filter,
                                                                               self.target_variable_activity,
                                                                               quantile, max_n_rows, **kwargs)
            at = pull_common_activities_by_activation_count(self.cel_analysis, act_string, extract_filter, due_date, 
                                                            **kwargs)
            at_count = get_number_times_activites_performed(self.cel_analysis, PQL(self.case_table_filters), act_unique, due_date,
                                                            reaction_time)
            at_encoded, self.training_settings = one_hot_encode_activities_by_activation_count(at, at_count,
                                                                       self.purpose,
                                                                       self.training_settings,
                                                                       self.training_settings_file,
                                                                       self.activities_final_file,
                                                                       self.signature)

        elif event_count:
            at_encoded = one_hot_encode_activities_by_event_count(self.cel_analysis, event_count)
        else:
            at = pull_common_activities(self.cel_analysis, extract_filter, due_date, **kwargs)
            at_processed = at.pivot(index="_CASE_KEY", columns='name', values="TIME_TO_DUE_DATE")
            return at_processed
        assert len(set(at_encoded)) == at_encoded.shape[1]
        
        self._logger.info(f'Obtaining data for {self.purpose} with activity table of shape: {at_encoded.shape}')
        return at_encoded


    def clean_data(self, raw_activities):
        return 1


    def one_hot_encode_activity_table(self, activities):
        return 1


    def obtain_activity_table(self, reaction_time=None, due_date=None):
        raw_activities = self.pull_activity_table(reaction_time=reaction_time, due_date=due_date)
        clean_activities = self.clean_data(raw_activities)
        encoded_activities = self.one_hot_encode_activity_table(clean_activities)
        return encoded_activities
