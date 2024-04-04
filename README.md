# esun_2023

## 隊伍名稱：TEAM_4010

## 作者：harryhowiefish
 
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