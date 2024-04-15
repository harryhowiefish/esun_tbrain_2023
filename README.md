# esun AI CUP 2023

## 競賽基本資料
#### 競賽題目：信用卡冒用偵測 
#### 隊伍名稱：TEAM_4010
#### 訓練集資料量：(8688526, 26)
#### 公開測試集資料量：609040 筆
#### 公開測試集資料量：805721 筆
#### 
#### 名次：第四名
#### 成果：F1-score 0.8457
[比賽說明連結](https://tbrain.trendmicro.com.tw/Competitions/Details/31)

## Python Packages used
numpy, pandas, catboost, scikit-learn

## 技術說明
- data cleaning
    - missing data only occur on categorical data -> fill na with -1 (new category)
- feature engineering
    - datetime to hour, minute, second categories
    - split time into time_of_day [morning, evening, night, midnight] categories
    - high transaction flag
    - day of week
    - transaction outside of country flag
    - concat "time of day" and "day of week"
    - concat 交易類別 and 交易型態
    - Overall Aggregate 
        - percentage of '商戶類別代碼', '消費城市', '收單行代碼', '特店代號'
    - By Card ID Aggregate
        - weekly avg transaction
        - fraud percentage (week to week)
        - average transaction amount (week to week)
        - percentage of '時段', '網路交易註記', '商戶類別代碼', '消費城市', '收單行代碼', '特店代號' (week to week)
        - new customer flag (week to week)
        - 網路消費習慣, 國內外消費習慣 flag (0 or 1) (week to week)
        - 國內消費比例, 消費頻率 (week to week)
- model design
    - param
        - learning_rate=0.4
        - depth=16
        - loss_function='Logloss'
        - class_weights=[1,220]
        - task_type='GPU'
        - l2_leaf_reg=7
        - eval_metric = F1
    - build for models with different sets of feature to emphasize different aspect of the data.
    - voting
        - average the probability from four models.

- key concept
    - Add in data week by week to feature engineering pipeline in order to preserve observation on new customer behavior in historical data.

## 檔案用途:
- Preprocess/: 存放前處理的code
- Model/: 存放模型相關code（catboost with GPU）
- Voting/:存放平均結果+輸出submission的code
- requirements.txt: 需要的套件
- main.py: 執行整個訓練流程

## 執行流程:
```
# 安裝所需套件
$ pip install -r requirements.txt 

# 執行資料前處理
$ python ./Preprocess/df_formatting_and_extract.py

# training,inference and voting
$ python main.py
```
## Action list
- add testing
- create some pipeline framework
- rewrite in spark
- create OLAP concept
- better readme
- attach a presentation file