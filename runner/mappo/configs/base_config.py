import argparse


class BaseConfig():
    """
    The configuration parser for common hyperparameters of all environment. 
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch 
        --cuda
            by default True, will use GPU to train; or else will use CPU; 
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.
    
    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or else will use concatenated local obs.
    
    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer. 
    
    Network parameters:
        --share_policy
            by default True, all agents will share the same network; set to make training agents use different policies. 
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards. 
        --use_valuenorm
            by default True, use running mean and std to normalize rewards. 
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs. 
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.
    
    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)
    
    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss 
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm 
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.  
        --huber_delta <float>
            coefficient of huber loss.  
    
    PPG parameters:
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)
    
    Run parameters:
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate
    
    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.
    
    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.
    
    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.
    
    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    
    def __init__(self) -> None:
        self.algorithm_name = 'mappo'  # "rmappo", "mappo"
        self.experiment_name = 'check'
        self.seed = int(1)
        self.cuda = True
        self.cuda_deterministic = True
        self.n_training_threads = int(1)
        self.n_rollout_threads = int(32)
        self.n_eval_rollout_threads = int(1)
        self.n_render_rollout_threads = int(1)
        self.num_env_steps = int(10e6)
        self.user_name = 'marl'
        self.use_wandb = True
        # env parameters
        self.env_name = 'StarCraft2'
        self.use_obs_instead_of_state = False
        # replay buffer parameters
        self.episode_length = int(200)
        # network parameters
        self.share_policy = True
        self.use_centralized_V = True
        self.stacked_frames = int(1)
        self.use_stacked_frames = False
        self.hidden_size = int(64)
        self.layer_N = int(1)
        self.use_ReLU = True
        self.use_popart = False
        self.use_valuenorm = True
        self.use_feature_normalization = True
        self.use_orthogonal = True
        self.gain = float(0.01)

        # recurrent parameters
        self.use_naive_recurrent_policy = False
        self.use_recurrent_policy = True
        self.recurrent_N = int(1)
        self.data_chunk_length = int(10)

        # optimizer parameters
        self.lr = float(5e-4)
        self.critic_lr = float(5e-4)
        self.opti_eps = float(1e-5)
        self.weight_decay = float(0)

        # ppo parameters
        self.ppo_epoch = int(15)
        self.use_clipped_value_loss = True
        self.clip_param = float(0.2)
        self.num_mini_batch = int(1)
        self.entropy_coef = float(0.01)
        self.value_loss_coef = float(1)
        self.use_max_grad_norm = True
        self.max_grad_norm = float(10.0)
        self.use_gae = True
        self.gamma = float(0.99)
        self.gae_lambda = float(0.95)
        self.use_proper_time_limits = False
        self.use_huber_loss = True
        self.use_value_active_masks = True
        self.use_policy_active_masks = True
        self.huber_delta = float(10.0)

        # run parameters
        self.use_linear_lr_decay = False
        # save parameters
        self.save_interval = int(1)

        # log parameters
        self.log_interval = int(5)

        # eval parameters
        self.use_eval = False
        self.eval_interval = int(25)
        self.eval_episodes = int(32)
        
        # render parameters
        self.save_gifs = False
        self.use_render = False
        self.render_episodes = int(5)
        self.ifi = float(0.1)

        # pretrained parameters
        self.model_dir = None
