"""
用途
- 更改欄位標題且修改資料型態來降低資料大小
- feature engineering

input
-training.csv
-public_processed.csv
-private_1_processed.csv

output
-train_data_extracted.csv
-val_data_extracted.csv
-public_data_extracted.csv
-private1_data_extracted.csv

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
tqdm.pandas()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def column_rename(df: pd.DataFrame) -> list:
    """
    用途
    - 更改欄位標題(改為中文以便後續使用)
    
    input
    - df(DataFrame)

    output
    - list: column name

    """
    col_dict = {'txkey':'交易序號','locdt':'授權日期','loctm':'授權時間',
                'chid':'顧客ID','cano':'交易卡號','contp':'交易類別','etymd':'交易型態',
                'mchno':'特店代號','acqic':'收單行代碼','mcc':'商戶類別代碼','conam':'轉換後交易金額',
                'ecfg':'網路交易註記','insfg':'分期交易註記','iterm':'分期期數','bnsfg':'是否紅利交易',
                'flam1':'實付金額','stocn':'消費地國別','scity':'消費城市','stscd':'狀態碼',
                'ovrlt':'超額註記碼','flbmk':'Fallback註記','hcefg':'支付型態','csmcu':'消費地幣別',
                'csmam':'消費地金額','flg_3dsmk':'3DS註記','label':'盜刷註記'}
    return [col_dict[col] for col in df.columns]

def extract_time(df:pd.DataFrame) -> pd.DataFrame:

    """
    用途
    - 把授權時間拆開成小時、分鐘、秒
    
    input
    - df(DataFrame)

    output
    - DataFrame
    """

    temp = df['授權時間'].apply(lambda x: str(x).zfill(6))
    df['授權小時'] = temp.apply(lambda x:int(str(x)[:2])).astype('int8')
    df['授權分鐘'] = temp.apply(lambda x:int(str(x)[2:4])).astype('int8')
    df['授權秒'] = temp.apply(lambda x:int(str(x)[4:])).astype('int8')
    df.drop('授權時間',axis=1,inplace=True)
    return df

def reformat(df:pd.DataFrame) -> pd.DataFrame:
    """
    用途
    - 修改資料型態來降低資料大小
    
    input
    - df(DataFrame)

    output
    - DataFrame
    """
    bool_col = ['網路交易註記', '分期交易註記', '是否紅利交易', '超額註記碼', 'Fallback註記', '3DS註記']
    cat_col = ['顧客ID', '交易卡號', '授權日期','授權小時','授權分鐘','授權秒', '交易類別', '交易型態', '特店代號', '收單行代碼',
                '商戶類別代碼', '分期期數', '消費地國別', '消費城市', '狀態碼', '支付型態', '消費地幣別']
    num_col = ['轉換後交易金額', '實付金額', '消費地金額']
    obj_col = [col for col in cat_col if df[col].dtype!='object']
    for col in obj_col:
        max_val = df[col].max()
        if max_val>10000:
            df[col] = df[col].astype('int32')
        elif max_val>100:
            df[col] = df[col].astype('int16')
        else:
            df[col] = df[col].astype('int8')
    df[bool_col] = df[bool_col].astype('int8')
    df[num_col] = df[num_col].astype('int32')
    return df

def create_date_list(all_date,size=4):
    """
    用途
    - 將日期拆成4天為一組的區間，以便後續資料處理
    
    input
    - all_date(list):所有日期

    output
    - list:切分後的sublist
    """
    sublists = [all_date[i:i + size] for i in range(0, len(all_date), size)]
    # Adding the last sublist with 5 elements
    if len(sublists[-1])<=2:
        sublists[-2] = sublists[-2]+sublists[-1]
        sublists.pop(-1)
    return sublists

class CardCycleProcessor:
    """
    用途
    - 每週期載入資料來針對特定週期做特定的統計分析，包含選擇性的資料後處理
    """

    def __init__(self, df: pd.DataFrame):
        """
        初始化class

        input:
        - df(DataFrame)
        """
        self.df = df

    def card_cycle_processing(self, date_range: list, col: str, method: str, name: str, 
                              post_processing=None, fillna: bool = True) -> pd.DataFrame:
        """
        依照特定週期載入資料來針對特定週期做特定的統計分析，包含選擇性的資料後處理

        input:
        - date_range (list): 日期週期(sublists)
        - col (str): 特定欄位
        - method (str): 統計方法(mean, median, count, sum, or ratio).　ratio等同value_counts(normalize=True)
        - name (str): 新的欄位名稱
        - post_processing (Callable, optional): 選擇性的資料後處理. Defaults to None.
        - fillna (bool): 填補缺失值

        output:
        - DataFrame
        """
        previous_dates = date_range[0].copy()
        for i in tqdm(range(1, len(date_range)), desc=name):
            if method == 'ratio':
                mapping = self.df[self.df['授權日期'].isin(previous_dates)].groupby('交易卡號')[col].value_counts(normalize=True).to_dict()
            else:
                mapping = self.df[self.df['授權日期'].isin(previous_dates)].groupby('交易卡號')[col].agg(method).to_dict()

            if post_processing is None:
                self.df.loc[self.df['授權日期'].isin(date_range[i]), name] = self.df[self.df['授權日期'].isin(date_range[i])]['交易卡號'].map(mapping)
            else:
                self.df.loc[self.df['授權日期'].isin(date_range[i]), name] = self.df[self.df['授權日期'].isin(date_range[i])].apply(lambda x: post_processing(x, mapping, col), axis=1)

            previous_dates += date_range[i]

        if fillna:
            self.df[name].fillna(0, inplace=True)

        return self.df

    @staticmethod
    def multi_key_mapping(x, mapping, col):
        """
        用來multi-key的dictionary map到資料表上的交易卡號以及特定欄位

        input:
        - x (pd.Series): 一行資料
        - mapping (dict): Multi-key dictionary
        - col (str): 特定欄位名稱

        output:
        - value
        """
        return mapping.get((x['交易卡號'], x[col]), 0)

    @staticmethod
    def spending_ratio(x, mapping, col):
        """
        計算當次消費和同卡號平均消費金額的倍率

        input:
        - x (pd.Series): 一行資料
        - mapping (dict): Single-key dictionary
        - col (str): 特定欄位名稱

        output:
        - value
        """
        if x['交易卡號'] not in mapping or mapping[x['交易卡號']]==0:
            return 0
        else:
            return x[col] / mapping[x['交易卡號']]

def update(x,col,mapping,method):
    """
    更新card_df裡特定卡號的數值

    input:
    - x (pd.Series): 一行資料
    - mapping (dict): Single-key dictionary
    - col (str): 特定欄位名稱

    output:
    - value
    """
    if x['交易卡號'] not in mapping:
        return x[col]
    else:
        if method=='sum':
            return x[col]+mapping[x['交易卡號']]
        elif method=='count':
            return x[col]+1

def check_behavior(x,reference_df,card_list,behavior_col,data_col):
    """
    檢查當次消費行為是否和行為習慣吻合 e.x.若主要網路消費，當次也是網路消費則為1，非網路消費則為0

    input:
    - x (pd.Series): 一行資料
    - reference_df (pd.DataFrame): 各卡號的行為偏好
    - card_list: 已經有消費紀錄的卡號list
    - behavior_col (str): 在reference_df裡的特定欄位
    - data_col(str): 在all_data裡的特定欄位

    output:
    - value
    """
    if x['交易卡號'] not in card_list:
        return -1
    label = reference_df.loc[x['交易卡號'],behavior_col]
    if label==-1:
        return -1
    elif label == x[data_col]:
        return 1
    elif label != x[data_col]:
        return 0


def query_ratio(x,reference_df,card_list,col):
    """
    索引特定的行為比例 e.x.國內消費比例, 消費頻率

    input:
    - x (pd.Series): 一行資料
    - reference_df (pd.DataFrame): 各卡號的行為偏好
    - card_list: 已經有消費紀錄的卡號list
    - col (str): 在reference_df裡的特定欄位

    output:
    - value
    """
    if x['交易卡號'] not in card_list:
        return 1
    return reference_df.loc[x['交易卡號'],col]

def main():
    print('loading data...')
    train = pd.read_csv('./dataset_1st/training.csv')
    public_test = pd.read_csv('./dataset_1st/public_processed.csv')
    private_test_1 = pd.read_csv('./dataset_2nd/private_1_processed.csv')

    all_data = pd.concat([train,public_test,private_test_1])
    train.columns = column_rename(train)
    all_data.columns = column_rename(all_data)
    all_data = extract_time(all_data)
    all_data.sort_values(['授權日期','授權小時','授權分鐘','授權秒','交易卡號'],inplace=True)
    all_data.reset_index(drop=True,inplace=True)
    all_data.fillna(-1,inplace=True)
    all_data = reformat(all_data)
    
    print('feature enginneering...')
    #金額超過正常刷卡金額的PR80只有用train data以避免data leakage
    all_data['高金額'] = (all_data['轉換後交易金額']>train[train['盜刷註記']==0]['轉換後交易金額'].quantile(q=.80)).astype('int8')
    del train,public_test,private_test_1

    #date_list在後續feature engineering的時候可以避免data leakage
    date_list = create_date_list(sorted(all_data['授權日期'].unique()))

    #用date_list做出週期
    all_data['授權週數'] = all_data['授權日期'] // 7
    all_data['授權週日'] = all_data['授權日期'] % 7
    # all_data['4_day_cycle'] = all_data['授權日期'] // 4
    # all_data['4_day_count'] = all_data['授權日期'] % 4
    for idx,sublist in enumerate(date_list):
        all_data.loc[all_data['授權日期'].isin(sublist),'loading_cycle'] = idx
    all_data['loading_cycle'] =all_data['loading_cycle'].astype('int8')

    all_data.loc[all_data['授權小時'].between(6,13),'時段'] = '早上'
    all_data.loc[all_data['授權小時'].between(13,18),'時段'] = '下午'
    all_data.loc[all_data['授權小時'].between(18,23),'時段'] = '晚上'
    all_data['時段'].fillna('凌晨',inplace=True)

    all_data['是否為國外消費'] = all_data['消費地國別']!=0

    #某些資料可以被視為numeric or categorical data
    for col in ['授權日期','授權週數','授權週日']:#,'4_day_cycle']:
        all_data[f"num_{col}"] = all_data[col]

    #組合特定cat_col
    all_data['授權週日_時段'] = all_data['授權週日'].astype('str')+"_"+all_data['時段']
    all_data['交易類別_交易型態'] = all_data['交易類別'].astype('str')+"_"+all_data['交易型態'].astype('str')



    #整體資料在各分類的佔比
    previous_dates = date_list[0].copy()
    for col in ['商戶類別代碼','消費城市','收單行代碼','特店代號']:
        for i in tqdm(range(1,len(date_list)),desc=f'{col}整體資料佔比'):
            mapping = all_data[all_data['授權日期'].isin(previous_dates)][col].value_counts(normalize=True).to_dict()
            all_data.loc[all_data['授權日期'].isin(date_list[i]),f'{col}消費總比例'] = all_data[all_data['授權日期'].isin(date_list[i])][col].map(mapping)
            previous_dates += date_list[i]  
        all_data[f'{col}消費總比例'].fillna(0)


    #計算同卡號(過去)每週平均刷卡次數，由於mapping計算方法與其他不同，因此沒有做成function
    previous_dates = date_list[0].copy()
    for i in tqdm(range(1,len(date_list)),desc='weekly_刷卡_count'):
        mapping = all_data[all_data['授權日期'].isin(previous_dates)].groupby(['交易卡號','4_day_cycle'])['盜刷註記'].count().groupby(level=0).mean().to_dict()
        all_data.loc[all_data['授權日期'].isin(date_list[i]),'weekly_刷卡_count']=all_data[all_data['授權日期'].isin(date_list[i])]['交易卡號'].map(mapping)
        previous_dates+=date_list[1]
    all_data['weekly_刷卡_count'].fillna(0,inplace=True)
    processor = CardCycleProcessor(all_data)

    all_data = processor.card_cycle_processing(date_list,'盜刷註記','mean','盜刷_mean')
    all_data = processor.card_cycle_processing(date_list,'轉換後交易金額','mean','個人平均消費金額倍率',CardCycleProcessor.spending_ratio)
    all_data = processor.card_cycle_processing(date_list,'轉換後交易金額','median','個人消費金額中位數倍率',CardCycleProcessor.spending_ratio)
    for col in ['時段','網路交易註記','商戶類別代碼','消費城市','收單行代碼','特店代號']:
        all_data = processor.card_cycle_processing(date_list,col,'ratio',f'卡號在{col}的比例',CardCycleProcessor.multi_key_mapping)
    
    #標註是否為新消費者(前面的cycle沒有看到的卡號)
    card_list = set()
    for i in tqdm(range(0,len(date_list)),desc='新消費者'):
        ##update all_data info first
        all_data.loc[all_data['授權日期'].isin(date_list[i]),'新消費者'] = all_data[all_data['授權日期'].isin(date_list[i])].apply(lambda x:x['交易卡號'] not in card_list,axis=1)
        #update card_df
        new_data = all_data[all_data['授權日期'].isin(date_list[i])]
        new_cards = [card for card in new_data['交易卡號'].unique() if card not in card_list]
        card_list.update(new_cards)

    #針對同卡號分析行為模式，包含網路消費習慣, 國內外消費習慣, 國內消費比例, 消費頻率並記錄在card_df
    print('running card behavior...this will take a while')
    card_df = pd.DataFrame(columns=['交易卡號','交易次數','網路消費次數','國內消費次數','交易週數','可消費週數','網路交易behavior','國內消費behavior'])
    card_list = set()
    for i in tqdm(range(0,len(date_list))):
        ##update all_data info first
        card_df.set_index('交易卡號',inplace=True)
        all_data.loc[all_data['授權日期'].isin(date_list[i]),'是否符合網路消費習慣'] = all_data[all_data['授權日期'].isin(date_list[i])].apply(lambda x:check_behavior(x,card_df,card_list,'網路交易behavior','網路交易註記'),axis=1)
        all_data.loc[all_data['授權日期'].isin(date_list[i]),'是否符合國內外消費習慣'] = all_data[all_data['授權日期'].isin(date_list[i])].apply(lambda x:check_behavior(x,card_df,card_list,'國內消費behavior','消費地國別'),axis=1)
        all_data.loc[all_data['授權日期'].isin(date_list[i]),'國內消費比例'] = all_data[all_data['授權日期'].isin(date_list[i])].apply(lambda x:query_ratio(x,card_df,card_list,'國內消費比例'),axis=1)
        all_data.loc[all_data['授權日期'].isin(date_list[i]),'消費頻率'] = all_data[all_data['授權日期'].isin(date_list[i])].apply(lambda x:query_ratio(x,card_df,card_list,'消費頻率'),axis=1)
        card_df.reset_index(inplace=True)


        #update card_df
        new_data = all_data[all_data['授權日期'].isin(date_list[i])]
        new_cards = [card for card in new_data['交易卡號'].unique() if card not in card_list]
        card_df = pd.concat([card_df,pd.DataFrame(new_cards,columns=['交易卡號'])])
        card_df['網路交易behavior'].fillna(-1,inplace=True) #new/inconclusive
        card_df['國內消費behavior'].fillna(-1,inplace=True) #new/inconclusive
        card_df['可消費週數'].fillna(1,inplace=True)
        card_df.fillna(0,inplace=True)
        card_list.update(new_cards)


        mapping = new_data.groupby('交易卡號')['網路交易註記'].count().to_dict()
        card_df['交易次數'] = card_df.apply(lambda x: update(x,'交易次數',mapping,'sum'),axis=1)
        card_df['交易週數'] = card_df.apply(lambda x: update(x,'交易週數',mapping,'count'),axis=1)
        mapping = new_data.groupby('交易卡號')['網路交易註記'].sum().to_dict()
        card_df['網路消費次數'] = card_df.apply(lambda x: update(x,'網路消費次數',mapping,'sum'),axis=1)
        mapping = new_data[new_data['消費地國別']==0].groupby('交易卡號')['轉換後交易金額'].count().to_dict()
        card_df['國內消費次數'] = card_df.apply(lambda x: update(x,'國內消費次數',mapping,'sum'),axis=1)
        card_df['國內消費比例'] = card_df['國內消費次數']/card_df['交易次數']

        card_df.loc[card_df['網路消費次數']/card_df['交易次數']>0.7,'網路交易behavior']=1
        card_df.loc[card_df['網路消費次數']/card_df['交易次數']<0.3,'網路交易behavior']=0
        card_df.loc[card_df['國內消費比例']>0.7,'國內消費behavior']=0
        card_df.loc[card_df['國內消費比例']<0.3,'國內消費behavior']=1
        card_df['可消費週數'] = card_df['可消費週數']+1
        card_df['消費頻率'] = card_df['交易週數']/card_df['可消費週數']

    all_data['是否符合網路消費習慣'] = all_data['是否符合網路消費習慣'].astype('int8')
    all_data['是否符合國內外消費習慣'] = all_data['是否符合國內外消費習慣'].astype('int8')


    print('export results...')
    data_type = {col:all_data[col].dtype for col in all_data.columns}
    with open('data_type.pkl','wb') as f:
        pickle.dump(data_type,f)
    
    train = all_data[all_data['授權日期'].isin(range(0,52))]
    val = all_data[all_data['授權日期'].isin(range(52,56))]
    public_test = all_data[all_data['授權日期'].isin(range(56,60))]
    private_test_1 = all_data[all_data['授權日期'].isin(range(60,65))]

    train.to_csv('train_data_extracted.csv',index=False)
    val.to_csv('val_data_extracted.csv',index=False)
    public_test.to_csv('public_data_extracted.csv',index=False)
    private_test_1.to_csv('private1_data_extracted.csv',index=False)
    card_df.to_csv('card_df.csv',index=False)

if __name__ == '__main__':
    main()  