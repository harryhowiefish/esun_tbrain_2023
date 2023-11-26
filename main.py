"""
執行訓練流程
訓練四個模型之後把結果平均後輸出
"""

from Model import model_1,model_2,model_3,model_4
from Voting import vanilla_voting

def main():
    model_1.train()
    print('model 1 complete')
    model_2.train()
    print('model 2 complete')
    model_3.train()
    print('model 3 complete')
    model_4.train()
    print('model 4 complete')
    vanilla_voting.voting()
    print('submission file created')

if __name__ == '__main__':
    main()