"""
執行訓練流程
"""

from Model import model_1,model_2,model_3,model_4
from Voting import vanilla_voting

def main():
    model_1.train()
    # model_2.train()
    # model_3.train()
    # model_4.train()
    # vanilla_voting.voting()

if __name__ == '__main__':
    main()