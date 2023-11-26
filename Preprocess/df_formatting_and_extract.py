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
    - dataframe

    output
    - list（column name)

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
    - dataframe

    output
    - dataframe
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
    - dataframe

    output
    - dataframe
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
    - list(所有日期)

    output
    - list(切分後的sublist)
    """
    sublists = [all_date[i:i + size] for i in range(0, len(all_date), size)]
    # Adding the last sublist with 5 elements
    if len(sublists[-1])<=2:
        sublists[-2] = sublists[-2]+sublists[-1]
    sublists.pop(-1)
    return sublists

class CardCycleProcessor:
    """
    A class for processing card cycle data within specified date ranges.

    This class provides functionalities to aggregate data over given date ranges
    and apply statistical methods on a specified column, with optional post-processing.

    Attributes:
        df (pd.DataFrame): A pandas DataFrame containing the data to be processed.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the CardCycleProcessor with a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to process.
        """
        self.df = df

    def card_cycle_processing(self, date_range: list, col: str, method: str, name: str, 
                              post_processing=None, fillna: bool = True) -> pd.DataFrame:
        """
        Process card cycle data in batches and apply statistical methods to a specified column.

        Args:
            date_range (list): List of date ranges (sublists) to process.
            col (str): Column name to be aggregated.
            method (str): Aggregation method (mean, median, count, sum, or ratio).
            name (str): Name of the new column to store the results.
            post_processing (Callable, optional): Function for additional processing. Defaults to None.
            fillna (bool): Fill NA values with 0 if True. Defaults to True.

        Returns:
            pd.DataFrame: The processed DataFrame with the new column added.
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
        Map a multi-key tuple from a DataFrame row to a value in the provided mapping.

        Args:
            x (pd.Series): A row of the DataFrame.
            mapping (dict): A dictionary mapping keys to values.
            col (str): Column name to be used as part of the key.

        Returns:
            The value from the mapping for the given key, or 0 if the key is not found.
        """
        return mapping.get((x['交易卡號'], x[col]), 0)

    @staticmethod
    def spending_ratio(x, mapping, col):
        """
        Calculate the spending ratio of a DataFrame row based on the provided mapping.

        Args:
            x (pd.Series): A row of the DataFrame.
            mapping (dict): A dictionary mapping keys to aggregate values.
            col (str): Column name whose value is to be divided by the aggregate value.

        Returns:
            The spending ratio, or 0 if the key is not in the mapping or the aggregate value is 0.
        """
        card_val = mapping.get(x['交易卡號'])
        if not card_val:
            return 0
        else:
            return x[col] / card_val

class BehaviorCycleProcessor:
    """
    A class for processing behavior cycles of card transactions.

    This class analyzes transaction data to update counts and assess specific habits based on transaction history.
    """

    def __init__(self, all_data: pd.DataFrame, date_list: list):
        """
        Initialize the BehaviorCycleProcessor with transaction data and date ranges.

        Args:
            all_data (pd.DataFrame): DataFrame containing transaction data.
            date_list (list): List of date ranges for processing.
        """
        self.all_data = all_data
        self.date_list = date_list
        self.card_df = pd.DataFrame(columns=['交易卡號'])
        

    def process_cycles(self):
        """
        Process the behavior cycles for the given date ranges and update the DataFrame.

        Iterates through the date ranges, updating transaction and internet consumption
        counts, and determining habitual internet consumption patterns.
        """
        self.card_list = set()
        for i in tqdm(range(0, len(self.date_list))):
            self.card_df.set_index('交易卡號', inplace=True)
            self.all_data.loc[self.all_data['授權日期'].isin(self.date_list[i]), '是否符合網路消費習慣'] = \
                self.all_data[self.all_data['授權日期'].isin(self.date_list[i])].apply(
                    lambda x: self.check_channel_matching(x, self.card_df, self.card_list), axis=1)
            self.card_df.reset_index(inplace=True)

            new_data = self.all_data[self.all_data['授權日期'].isin(self.date_list[i])]
            new_cards = [card for card in new_data['交易卡號'].unique() if card not in self.card_list]
            self.card_df = pd.concat([self.card_df, pd.DataFrame(new_cards, columns=['交易卡號'])])
            self.card_df['習慣網路消費label'] = -1  # inconclusive
            self.card_df.fillna(0, inplace=True)
            self.card_list.update(new_cards)

            mapping = new_data.groupby('交易卡號')['轉換後交易金額'].count().to_dict()
            self.card_df['交易次數'] = self.card_df.apply(lambda x: self.update(x, '交易次數', mapping), axis=1)
            mapping = new_data.groupby('交易卡號')['網路交易註記'].sum().to_dict()
            self.card_df['網路消費次數'] = self.card_df.apply(lambda x: self.update(x, '網路消費次數', mapping), axis=1)
            self.card_df.loc[self.card_df['網路消費次數'] / self.card_df['交易次數'] > 0.7, '習慣網路消費label'] = 1
            self.card_df.loc[self.card_df['網路消費次數'] / self.card_df['交易次數'] < 0.3, '習慣網路消費label'] = 0


    @staticmethod
    def update(x, col, mapping,method):
        """
        Update the count for a card in the DataFrame.

        Args:
            x (pd.Series): A row of the DataFrame.
            col (str): Column name to be updated.
            mapping (dict): A dictionary mapping card numbers to new count values.

        Returns:
            Updated value for the column.
        """
        if x['交易卡號'] not in mapping:
            return x[col]
        else:
            if method=='sum':
                return x[col]+mapping[x['交易卡號']]
            elif method=='count':
                return x[col]+1

    @staticmethod
    def check_behavior(x, reference_df):
        """
        Check if a transaction matches the habitual internet consumption pattern.

        Args:
            x (pd.Series): A row of the DataFrame.
            reference_df (pd.DataFrame): DataFrame containing reference labels.

        Returns:
            An integer indicating the match status (-1, 0, or 1).
        """

        reference = reference_df.loc[x['交易卡號']]
        if len(reference)==0:
            return -1
        if reference['習慣網路消費label'] == -1:
            return -1
        elif reference['習慣網路消費label'] == x['網路交易註記']:
            return 1
        elif reference['習慣網路消費label'] != x['網路交易註記']:
            return 0

def update(x,col,mapping,method):
    if x['交易卡號'] not in mapping:
        return x[col]
    else:
        if method=='sum':
            return x[col]+mapping[x['交易卡號']]
        elif method=='count':
            return x[col]+1

def check_behavior(x,reference_df,card_list,behavior_col,data_col):
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
    if x['交易卡號'] not in card_list:
        return 1
    return reference_df.loc[x['交易卡號'],col]

def main():
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

    #金額超過正常刷卡金額的PR80只有用train data以避免data leakage
    all_data['高金額'] = all_data['轉換後交易金額']>train[train['盜刷註記']==0]['轉換後交易金額'].quantile(q=.80)
    del train,public_test,private_test_1

    #4 day cycle為了配合資料提供週期而定
    all_data['授權週數'] = all_data['授權日期'] // 7
    all_data['授權週日'] = all_data['授權日期'] % 7
    all_data['4_day_cycle'] = all_data['授權日期'] // 4
    all_data['4_day_count'] = all_data['授權日期'] % 4

    all_data.loc[all_data['授權小時'].between(6,13),'時段'] = '早上'
    all_data.loc[all_data['授權小時'].between(13,18),'時段'] = '下午'
    all_data.loc[all_data['授權小時'].between(18,23),'時段'] = '晚上'
    all_data['時段'].fillna('凌晨')

    all_data['是否為國外消費'] = all_data['消費地國別']!=0

    #某些資料可以被視為numeric or categorical data
    for col in ['授權日期','授權週數','授權週日','4_day_cycle']:
        all_data[f"num_{col}"] = all_data[col]

    #組合特定cat_col
    all_data['授權週日_時段'] = all_data['授權週日'].astype('str')+all_data['時段']
    all_data['交易類別_交易型態'] = all_data['交易類別'].astype('str')+all_data['交易型態'].astype('str')

    #date_list在後續feature engineering的時候可以避免data leakage
    date_list = create_date_list(sorted(all_data['授權日期'].unique()))

    #整體資料在各分類的佔比
    previous_dates = date_list[0].copy()
    for i in tqdm(range(1,len(date_list)),desc='整體資料佔比'):
        for col in ['商戶類別代碼','消費城市','收單行代碼','特店代號']:
            mapping = all_data[all_data['授權日期'].isin(previous_dates)][col].value_counts(normalize=True).to_dict()
            all_data.loc[all_data['授權日期'].isin(date_list[i]),f'{col}消費總比例'] = all_data[all_data['授權日期'].isin(date_list[i])][col].map(mapping)
        previous_dates += date_list[i]  


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
    
    print('running card behavior...this will take a while')
    card_df = pd.DataFrame(columns=['交易卡號','交易次數','網路消費次數','國內消費次數','交易週數','可消費週數'])
    card_list = set()
    for i in tqdm(range(0,len(date_list))):
        ##update all_data info first
        card_df.set_index('交易卡號',inplace=True)
        all_data.loc[all_data['授權日期'].isin(date_list[i]),'是否符合網路消費習慣'] = all_data[all_data['授權日期'].isin(date_list[i])].apply(lambda x:check_behavior(x,card_df,card_list,'網路交易behavior','網路交易註記'),axis=1)
        all_data.loc[all_data['授權日期'].isin(date_list[i]),'是否符合國內外消費習慣'] = all_data[all_data['授權日期'].isin(date_list[i])].apply(lambda x:check_behavior(x,card_df,card_list,'國內消費為主','消費地國別'),axis=1)
        all_data.loc[all_data['授權日期'].isin(date_list[i]),'國內消費比例'] = all_data[all_data['授權日期'].isin(date_list[i])].apply(lambda x:query_ratio(x,card_df,card_list,'國內消費比例'),axis=1)
        all_data.loc[all_data['授權日期'].isin(date_list[i]),'消費頻率'] = all_data[all_data['授權日期'].isin(date_list[i])].apply(lambda x:query_ratio(x,card_df,card_list,'消費頻率'),axis=1)
        card_df.reset_index(inplace=True)


        #update card_df
        new_data = all_data[all_data['授權日期'].isin(date_list[i])]
        new_cards = [card for card in new_data['交易卡號'].unique() if card not in card_list]
        card_df = pd.concat([card_df,pd.DataFrame(new_cards,columns=['交易卡號'])])
        card_df['網路交易behavior'] = -1 #inconclusive
        card_df['國內消費為主'] = -1 #inconclusive
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

        card_df.loc[card_df['網路消費次數']/card_df['交易次數']>0.7,'習慣網路消費label']=1
        card_df.loc[card_df['網路消費次數']/card_df['交易次數']<0.3,'習慣網路消費label']=0
        card_df.loc[card_df['國內消費比例']>0.7,'國內消費為主']=0
        card_df.loc[card_df['國內消費比例']<0.3,'國內消費為主']=1
        card_df['可消費週數'] = card_df['可消費週數']+1
        card_df['消費頻率'] = card_df['交易週數']/card_df['可消費週數']


    print('export results...')
    data_type = {col:all_data[col].dtype for col in all_data.columns}
    with open('data_type.pkl','wb') as f:
        pickle.dump(data_type,f)
    
    train = all_data[all_data['授權日期'].isin(range(0,52))]
    val = all_data[all_data['授權日期'].isin(range(52,56))]
    public_test = all_data[all_data['授權日期'].isin(range(56,60))]
    private_test_1 = all_data[all_data['授權日期'].isin(range(61,65))]

    train.to_csv('train_data_extracted.csv',index=False)
    val.to_csv('val_data_extracted.csv',index=False)
    public_test.to_csv('public_data_extracted.csv',index=False)
    private_test_1.to_csv('private1_data_extracted.csv',index=False)

if __name__ == '__main__':
    main()  