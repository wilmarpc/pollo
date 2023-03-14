prediction_scoring = {
'Mean_abs': 'neg_mean_absolute_error',
'Mean_sqr': 'neg_mean_squared_error',
'Mean_log': 'neg_mean_squared_log_error'
}

classification_scoring = {
'AUC': 'roc_auc', 
'Brier': 'brier_score_loss',
'Recall': 'recall',
'Precision': 'precision',
'F1': 'f1',
'Accuracy': 'accuracy'
}

regression_scoring = {
'Explained_variance': 'explained_variance_score', 
'R2': 'r2_score',
'Mean_abs': 'neg_mean_absolute_error',
'Mean_sqr': 'neg_mean_squared_error'
}

