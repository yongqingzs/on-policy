from onpolicy.scripts.train.train_mpe_new import main as train_main
from onpolicy.scripts.render.render_mpe_new import main as render_main
from runner.mappo.configs.mpe_config import MPEConfig

TRAIN_ID = 'n'
ENV_ID = 'simple_spread'  # simple_adversary
ALGO_ID = 'rmappo'  # rmappo

if __name__ == '__main__':
    args = MPEConfig(ENV_ID, ALGO_ID)
    args.num_env_steps  = 2000

    if TRAIN_ID == 'y':
        # 改变参数
        train_main(args)
    elif TRAIN_ID == 'n':
        args.use_render = True  # render也会创建results
        args.cuda = False
        args.model_dir = f'./results/MPE/{ENV_ID}/{ALGO_ID}/check/run1/models'
        render_main(args)
    