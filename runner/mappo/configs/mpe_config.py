from runner.mappo.configs.base_config import BaseConfig

"""
来自train_mpe.parse_args
"""

class MPEConfig(BaseConfig):
    def __init__(self, scenario_name, algorithm_name) -> None:
        super().__init__()
        self.env_name = 'MPE'

        # 来自train_mpe.parse_args
        self.scenario_name = scenario_name
        self.num_landmarks = 3
        self.num_agents = 2
        self.algorithm_name = algorithm_name
        
        # 来自train_mpe_comm.sh
        self.n_training_threads = 1
        self.n_rollout_threads = 1  # 128
        self.num_mini_batch = 1
        self.episode_length = 25
        self.num_env_steps  = 2000000
        self.ppo_epoch = 15
        self.gain = 0.01
        self.lr = 7e-4
        self.critic_lr = 7e-4
        self.wandb_name = "xxx"
        self.use_wandb = False
        self.user_name = "cc"
        self.share_policy = True
        