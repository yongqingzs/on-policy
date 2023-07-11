from onpolicy.scripts.train.train_mpe_new import main
from runner.mappo.configs.mpe_config import MPEConfig

TRAIN_ID = 'y'
ENV_ID = 'simple_spread'  # simple_adversary

if __name__ == '__main__':
    args = MPEConfig(ENV_ID)
    
    if TRAIN_ID == 'y':
        # 改变参数
        main(args)
    