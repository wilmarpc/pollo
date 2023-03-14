import pandas as pd

def save_batch_to_dict(batch_results, model_name, mode, y_pred, y_true):
    res = {'y_true': y_true, f'y_pred_{model_name}': y_pred}
    if model_name in batch_results.keys():
        batch_results[mode][model_name].update(res)
    else:
        batch_results[mode][model_name] = res
    return batch_results

def batch_dict_to_df(mode, batch_result_dict, batch_result_df):
    models = list(batch_result_dict[mode].keys())
    batch_result_df[mode]['y_true'] = batch_result_dict[mode][models[0]]['y_true']
    for model_name in models:
        batch_result_df[mode]['y_pred_' + model_name] = (
            batch_result_dict[mode][model_name]['y_pred_' + model_name]
        )
    return batch_result_df