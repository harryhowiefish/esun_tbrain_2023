import pandas as pd
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score

def voting():
    result_1 = pd.read_csv('./result/01_baseline_with_public_retrain.csv',index_col=0)
    result_2 = pd.read_csv('./result/02_online_pattern_with_public_retrain.csv',index_col=0)
    result_3 = pd.read_csv('./result/03_country_pattern_with_public_retrain.csv',index_col=0)

    result = pd.concat([result_1,result_2,result_3],axis=1)
    result['pred_mean'] = result.mean(axis=1)
    result = result['pred_mean'].round().astype('int').to_frame()
    result.columns = ['pred']
    result.to_csv('./submission.csv',index=False)