import os
from pycelonis.pql import PQL, PQLColumn
from celonis_ml.data_preprocessing import DataLoader
import pandas as pd
import numpy as np
import logging
from scipy.stats import chi2_contingency


class FeatureRecommender:
    """
    Calculates Feature importance for the tables and
    columns available in the Datamodel, for each column it
    calculates a score between 0 and 1, with higher scores
    indicating that the KPI you are looking at is
    correlated with the column/feature you are looking at.

    Attributes
    ----------
    data_init : data_preprocessing.DataLoader
        the DataLoader instance which will controll the
        data gaterhing process of this trainer.
    tables_to_include : string or list of strings
        the names of the tables in the datamodel you want
        check.
    tables_to_exclude : string or list of strings
        the names of the tables in the dataframe that you
        want to exclude from the search

    """

    def __init__(self, cel_analysis, tables_to_include=[], tables_to_exclude=[], shared_selection_url=None):
        """
        Parameters
        ----------
        tables_to_include : string or list of strings
            the names of the tables in the datamodel you
            want check.
        tables_to_exclude : string or list of strings
            the names of the tables in the dataframe that you
            want to exclude from the search


        """
        self.cel_analysis = cel_analysis
        if shared_selection_url:
            self._variables_filters_query = self.cel_analysis.process_shared_selection_url(shared_selection_url)
        else:
            self._variables_filters_query = PQL(self.cel_analysis.published.calculate_variables())

        # setup working directory
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        self.tables_to_include = tables_to_include
        self.tables_to_exclude = tables_to_exclude
        self._logger = logging.getLogger(__name__)

    def input_features(self):
        """
        specifies the tables and/or columns that one wants
        to check feature importance for
        """

        # making sure the input tables are good
        if self.tables_to_include is not None:
            if isinstance(self.tables_to_include, (list,)):
                self.tables_to_include = [t.upper() for t in self.tables_to_include]
            elif isinstance(self.tables_to_include, str):
                self.tables_to_include = [self.tables_to_include.upper()]
            else:
                raise ValueError("tables_to_include must be None, a string or a list")

            tables_used = list(self.cel_analysis.datamodel.tables.names.keys())
            tables_used = [t.upper() for t in tables_used]

            self.tables_to_include = [table.replace('"', "") for table in self.tables_to_include]
            for table in self.tables_to_include:
                if table.replace('"', "") not in tables_used:
                    raise ValueError(
                        "the table: %s does not exist in the datamodel %s" % (table, self.cel_analysis.datamodel.name)
                    )

            return self.tables_to_include
        else:
            # making sure the input tables are good
            if self.tables_to_exclude is not None:
                if isinstance(self.tables_to_exclude, (list,)):
                    self.tables_to_exclude = [t.upper() for t in self.tables_to_exclude]
                elif isinstance(self.tables_to_exclude, str):
                    self.tables_to_exclude = [self.tables_to_exclude]
                else:
                    raise ValueError("tables_to_exclude must be None, a string or a list")

            tables_used = []
            # going through the tables of the datamodel
            for tab in list(self.cel_analysis.datamodel.tables.names.keys()):
                # select the tables that were not excluded
                if (
                    tab
                    not in (
                        [self.cel_analysis.datamodel.process_configurations[0].activity_table.name] + self.tables_to_exclude
                    )
                    and "Py_" not in tab
                ):
                    tables_used += [tab]
            return tables_used

    def cramers_corrected_stat(self, crosstab):
        """
        calculate Cramers V statistic for categorical-categorical association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
        To see the formula: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V#Bias_correction
        """

        chi2 = chi2_contingency(crosstab, correction=(len(crosstab) != 2))[0]
        n = crosstab.sum().sum()
        phi2 = chi2 / n
        r, k = crosstab.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    def cramers_corrected_stat_multiclass(self, x, y):
        """ Calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher, 
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        if "float" in x.dtype:
            x = x.astype(str)
        if "float" in y.dtype:
            y = y.astype(str)

        result = -1
        if len(x.value_counts()) == 1:
            print("First variable is constant")
        elif len(y.value_counts()) == 1:
            print("Second variable is constant")
        else:
            conf_matrix = pd.crosstab(x, y)

            if conf_matrix.shape[0] == 2:
                correct = False
            else:
                correct = True

            chi2 = ss.chi2_contingency(conf_matrix, correction=correct)[0]

            n = sum(conf_matrix.sum())
            phi2 = chi2 / n
            r, k = conf_matrix.shape
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)
            result = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
        return round(result, 6)

    def run(self, push=True, input_cases_sheet="Model Setup 5/5", input_cases_table="model_input_cases_table"):
        """
        runs the feature recommender and pushes table with
        recommendations to Celonis
        """
        self.data_init = DataLoader(self.cel_analysis)
        self.tables_used = self.input_features()
        ucq = self.data_init.get_use_case_query(input_cases_sheet, input_cases_table)
        self.target_variable_query = [c for c in ucq.columns if c.name.lower() == "target_variable"][0]
        try:
            self.classes = self.data_init.pull_data(
                PQL(
                    [PQLColumn(f"DISTINCT {self.target_variable_query.query}", "levels"), self._variables_filters_query]
                )
            ).squeeze()
            self.pred_type = "classification"
        except:
            self.pred_type = "regression"
            df = self.data_init.pull_data(
                PQL([PQLColumn(self.target_variable_query.query, "levels"), self._variables_filters_query])
            )
            self.classes = df.groupby(pd.cut(df["levels"].values, bins=20)).sum()
        feat_imp = self.feature_importance()
        if push:
            self.cel_analysis.datamodel.push_table(feat_imp, "Py_DDP_Feature_recommendation")
        return feat_imp

    def feature_importance(self, var_type="discrete"):
        """
        goes through the tables_used specified in the
        instance and checks for correlation with the
        target KPI

        Parameters
        ----------
        var_type: {'discrete', 'conitnuous'}
            type of the variable the importance is supposed
            to be measured. default is "discrete"
            For Classification purposes var_type should be
            "discrete",for Regression Purposes "continuous"

        push : bool
            Push results back to Celonis. True|False. = False)

        Returns
        ----------
        res: pandas DataFrame
            DataFrame with the results of the importance
            measure. Fields:
            ["table",
            "column",
            "column_type",
            "cramers_v"]
        """
        # TODO: for example take all features in the table as they are and then
        # do PCA and/or select k best for other features
        res = pd.DataFrame(columns=["table", "column", "column_type", "distinct_values", "cramers_v/abs(corr)"])

        for tab in self.cel_analysis.datamodel.tables:
            if tab.name in self.tables_used:
                # go through columns of the table
                for colx in tab.columns:
                    col_type = colx.get("type")
                    col = colx.get("name")
                    name = f"{tab.name}.{col}"
                    cel_name = f'"{tab.name}"."{col}"'
                    # exlude case keys
                    if col in (self.cel_analysis.datamodel.case_table_key + ["_CASE_KEY"]):
                        continue

                    query = PQL([PQLColumn(f"DISTINCT {cel_name}", name), self._variables_filters_query])

                    # exclude date columns
                    if col_type == "DATE":
                        continue
                    # calculate chi square for categorical columns
                    elif col_type == "STRING":
                        query += PQLColumn(
                            f"CASE WHEN ISNULL({cel_name})=1 THEN 'missing' ELSE {cel_name}||'' END", name
                        )

                    elif col_type == "FLOAT" or col_type == "INTEGER":
                        query += PQLColumn(f"CASE WHEN ISNULL({cel_name})=1 THEN 0||'' ELSE {cel_name}||'' END", name)

                    query += PQLColumn(
                        f"COUNT_TABLE({self.cel_analysis.datamodel.process_configurations[0].case_table.name})",
                        "class_size",
                    )

                    if self.pred_type == "regression":
                        for c in self.classes.index.values:
                            query += PQLColumn(
                                f"SUM (CASE WHEN ({self.target_variable_query.query}) < "
                                f"{c.index.values[0].right} AND {self.target_variable_query.query} >= "
                                f"{c.index.values[0].left} THEN 1 ELSE 0 END)",
                                f"count_class_{c}",
                            )
                    else:
                        for c in self.classes.values:
                            c2 = f"'{c}'" if isinstance(c, str) else c
                            query += PQLColumn(
                                f"SUM (CASE WHEN ({self.target_variable_query.query}) = {c2} THEN 1 ELSE 0 END)",
                                f"count_class_{c}",
                            )

                    df = self.data_init.pull_data(query)
                    if df.shape[0] > 20:
                        a = df[df.class_size < df.class_size.quantile(0.2)].sum()
                        a[name] = "123"
                        a.name = 22
                        df = df[df.class_size >= df.class_size.quantile(0.2)]
                        df = df.append(a, ignore_index=True)
                    if col_type != "STRING" and df.shape[0] > 100:
                        # If feature is numerical, we group the values and treat it as categorical
                        df[name] = df[name].astype("float64")
                        df = df.groupby(pd.cut(df[name].values, bins=20)).sum()
                    if df.shape[0] > 20:
                        df = df.nlargest(20, "class_size")
                    try:
                        crosstab = df[list(set(df.columns) - set([name, "class_size"]))]
                        cram_v = self.cramers_corrected_stat(crosstab)
                        self._logger.info(f"Cramers V for {tab.name}.{col}: {cram_v}")
                    except:
                        self._logger.info(f"Cramers V cannot be calculated for: {tab.name}.{col}")
                        cram_v = None
                    # add to the results table
                    res = res.append(
                        {
                            "table": tab.name,
                            "column": col,
                            "column_type": col_type,
                            "distinct_values": df.shape[0],
                            "cramers_v/abs(corr)": cram_v,
                        },
                        ignore_index=True,
                    )
                    res["distinct_values"] = res.distinct_values.astype(int)
        # save and push back feature importance results
        res = res.sort_values("cramers_v/abs(corr)", ascending=False)
        res = res.dropna()
        return res
