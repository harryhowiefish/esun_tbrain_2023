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


## 複賽筆記;

- 賽前準備：
  1. 跑pretrain model裡的4個ipynb建立model
  2. 跑pickle_everything.ipynb把需要的資料先存成pkl跟csv
- 比賽當天：
  1. 上傳prep資料夾內預先準備好的檔案
  2. 上傳4個with_public.cbm的pretrain model
  3. 執行process_new_data針對新的資料做特徵工程
  4. 執行pred_and_vote來輸出submission