"""
執行訓練流程
訓練四個模型之後把結果平均後輸出
"""

from Model import catboost
from Voting import vanilla_voting

def main():
    drop_col = ['盜刷註記','交易序號',
                '授權日期','授權週數','4_day_cycle','4_day_count',
                '個人消費金額中位數倍率',
                '個人平均消費金額倍率',
                '卡號在網路交易註記比例',
                '授權秒'
                '是否符合網路消費習慣','是否符合國內外消費習慣',
                '授權週日_時段','交易類別_交易型態',
                '新消費者','消費頻率'
                ]
    catboost.train(drop_col,'01_baseline_with_public_retrain.csv')
    print('model 1 complete')


    drop_col = ['盜刷註記','交易序號',
                '授權日期','授權週數','4_day_cycle','4_day_count',
                '個人消費金額中位數倍率',
                '個人平均消費金額倍率',
                '卡號在網路交易註記比例',
                '授權秒'
                '同卡號在特店代號的比例','時段',
                '是否符合國內外消費習慣',
                '授權週日_時段','交易類別_交易型態',
                '新消費者','消費頻率'
                ]
    catboost.train(drop_col,'02_online_pattern_with_public_retrain.csv')
    print('model 2 complete')

    drop_col = ['盜刷註記','交易序號',
                '授權日期','授權週數','4_day_cycle','4_day_count',
                '個人消費金額中位數倍率',
                '個人平均消費金額倍率',
                '卡號在網路交易註記比例',
                '授權秒',
                '消費地幣別',
                '是否符合網路消費習慣',
                '新消費者','消費頻率'
                ]
    catboost.train(drop_col,'03_country_pattern_with_public_retrain.csv')
    print('model 3 complete')
    
    drop_col = ['盜刷註記','交易序號',
                '授權日期','授權週數','4_day_cycle','4_day_count',
                '卡號在網路交易註記比例',
                'num_授權日期', 'num_授權週數',
                '授權秒',
                '消費地幣別','是否符合網路消費習慣',
                ]
    
    catboost.train(drop_col,'04_shopping_freq_with_public_retrain.csv')
    print('model 4 complete')


    vanilla_voting.voting()
    print('submission file created')

if __name__ == '__main__':
    main()