# 模型訓練
本資料夾內含一個程式:
- catboost.py: 用部分的feature進行訓練，並輸出預測的probability

## ML框架
- CatBoost ([GPU training is non-deterministic](https://catboost.ai/en/docs/features/training-on-gpu#) 因此每次的結果會有一些落差)
## 參數設定
- iterations=150
- learning_rate=0.4
- depth=16
- loss_function='Logloss'
- class_weights=[1,220]
- task_type='GPU'
- random_seed=5
- l2_leaf_reg=7
