"""
用途
- 用部分的feature進行訓練

input
- train_data_extracted.csv
- val_data_extracted.csv
- public_data_extracted.csv
- private1_data_extracted.csv

output
- DataFrame: submission prediction (probability)

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import catboost as cb
from catboost import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
import time
import os


def train(drop_col:list,filename:str):
    """
    用途
    - 用部分的feature進行訓練

    input
    - drop_col(list):要刪去哪些欄位
    - filename(str):輸出的檔名

    output
    - DataFrame: submission prediction (probability)

    """
    print('loading data...')
    train = pd.read_csv('./train_data_extracted.csv')
    val = pd.read_csv('./val_data_extracted.csv')
    public_test = pd.read_csv('./public_data_extracted.csv')
    private_test_1 = pd.read_csv('./private1_data_extracted.csv')
    tables = [train,val,public_test,private_test_1]

    with open('./data_type.pkl','rb') as f:
        data_type = pickle.load(f)
    for table in tables:
        table.fillna(0,inplace=True)
        for col in table.columns:
            table[col] = table[col].astype(data_type[col])
    
    public_ans = pd.read_csv('./dataset_2nd/public.csv')
    public_ans = public_ans[['txkey','label']]
    public_ans.set_index('txkey',inplace=True)
    gt = public_ans.loc[public_test.set_index('交易序號').index,'label']
    
    
    cat_col = ['顧客ID', '交易卡號',
            '授權日期','授權週數','4_day_cycle','4_day_count',
            '授權週日','時段',
            '授權小時','授權分鐘','授權秒', 
            '交易類別', '交易型態', '特店代號', '收單行代碼',
            '商戶類別代碼', '分期期數', '消費地國別', '消費城市', '狀態碼', '支付型態', '消費地幣別',
            '是否符合網路消費習慣','是否符合國內外消費習慣',
            '授權週日_時段','交易類別_交易型態',
            '新消費者'
                ]
    cat_col = [col for col in cat_col if col not in drop_col]
    selected_col = [col for col in train.columns if col not in drop_col]

    print('creating dataset ...')
    train_dataset = cb.Pool(pd.concat([train,val])[selected_col], pd.concat([train,val])['盜刷註記'],cat_features=cat_col) 
    val_dataset = cb.Pool(public_test[selected_col], gt,cat_features=cat_col) 
    time.sleep(10)
    print('training model without public testset ...')
    model = cb.CatBoostClassifier(iterations=150,learning_rate=0.4,depth=16,loss_function='Logloss',
                                class_weights=[1,220],
                                task_type='GPU',
                                random_seed=5,
                                l2_leaf_reg=7,
                                eval_metric=metrics.F1(use_weights=False)
                                ) #class weight 220 best
    model.fit(train_dataset,eval_set=val_dataset)
    
    
    inference_dataset = cb.Pool(public_test[selected_col],cat_features=cat_col)
    public_test['pred'] = model.predict_proba(inference_dataset)[:,1]
    print('f1:',f1_score(gt,public_test['pred'].round()))
    print('precision:',precision_score(gt,public_test['pred'].round()))
    print('recall:',recall_score(gt,public_test['pred'].round()))

    sub = pd.read_csv('./31_範例繳交檔案.csv')
    sub.set_index('txkey',inplace=True)
    sub.drop('pred',axis=1,inplace=True)
    sub.loc[public_test.set_index('交易序號').index,'pred'] = public_test.set_index('交易序號')['pred']

    # private_inference = cb.Pool(private_test_1[selected_col],cat_features=cat_col)
    # private_test_1['pred'] = model.predict_proba(private_inference)[:,1]
    # sub.loc[private_test_1.set_index('交易序號').index,'pred'] = private_test_1.set_index('交易序號')['pred']
    # sub['pred'].round().value_counts()
    print('training final model with public testset ...')
    train_dataset = cb.Pool(pd.concat([train,val,public_test])[selected_col], pd.concat([train['盜刷註記'],val['盜刷註記'],gt]),cat_features=cat_col) 
    time.sleep(10)
    model = cb.CatBoostClassifier(iterations=150,learning_rate=0.4,depth=16,loss_function='Logloss',
                                class_weights=[1,220],
                                task_type='GPU',
                                random_seed=5,
                                l2_leaf_reg=7,
                                eval_metric=metrics.F1(use_weights=False)
                                ) #class weight 220 best
    model.fit(train_dataset)

    private_inference = cb.Pool(private_test_1[selected_col],cat_features=cat_col)
    private_test_1['pred'] = model.predict_proba(private_inference)[:,1]
    sub.loc[private_test_1.set_index('交易序號').index,'pred'] = private_test_1.set_index('交易序號')['pred']
    
    print('all submission assigned',sub.isna().any()==False)

    print('saving result...')
    if not os.path.exists('result'):
        # Create the folder
        os.makedirs('result')
    sub.reset_index().to_csv(f'./result/{filename}',index=False)