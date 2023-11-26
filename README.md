# esun_2023

## 隊伍名稱：TEAM_4010

## 作者：harryhowiefish
 
## 檔案用途:
- Preprocess/: 存放前處理的code
- Model/: 存放模型相關code，包含
- Voting/:將4個模型的結果平均後輸出csv
- requirements.txt: 需要的套件
- main.py: 執行整個訓練流程

## 執行流程:
```
# 安裝所需套件
$ pip install -r requirements.txt 

# 執行資料前處理
$ python ./Preprocess/df_formatting_and_extract.py

# training,inference and voting
$ main.py
```
