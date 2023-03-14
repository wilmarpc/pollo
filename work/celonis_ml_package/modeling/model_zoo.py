"""Model Zoo for the Celonis Machine Learning Suite.
"""
import pandas as pd
from celonis_ml import modeling

MODEL_CLASSES = {
    'lgbm_classifier': modeling.LGBMClassifier,
    'rf_classifier': modeling.RFClassifier,
    'lr_classifier': modeling.LogisticRegressionClassifier,
    'xgb_classifier': modeling.XGBClassifier,
    'knn_classifier': modeling.KNNClassifier,
    'lgbm_regressor': modeling.LGBMRegressor
    #'deep_mlp_class': deepmodeling.MLPClassifier
}


class ModelZoo:
    """
    Abstraction layer where all parameters are specified for multiple
    available classification algorithms. This works as a wrapper in order
    to allow usage of many different pipelines indistinctively.

    Attributes
    ----------
    feature_importance : pandas DataFrame
        Dataframe containing the importance of the features
        will only be calculated for some modeling

    """
    def __init__(self):
        self.feature_importance_table = pd.DataFrame(
            columns=["feature_name", "feature_importance"])


    def reduce_to_one(self, model_name):
        """
        Reduce model zoo to a single model, specified by `model_name`
        and initialize its attributes.
        """
        pass
        
    @staticmethod
    def get_model_class(name):
        m = MODEL_CLASSES.get(name)
        if m is None:
            raise KeyError("Model %s not found." % name)
        return m

    @classmethod
    def load_model(self, model_name, model_path, **kwargs):
        model_clazz = self.get_model_class(model_name)
        model_instance = model_clazz.load_model(model_path, **kwargs)
        zoo_model = model_clazz()
        zoo_model.model_instance = model_instance
        return zoo_model
