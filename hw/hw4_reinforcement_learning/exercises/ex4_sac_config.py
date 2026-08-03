SAC_PARAMETERS = {
    "seed": 42,
    "hidden_sizes": [256, 128, 128],
    "total_iterations": 2048, # total number of training iterations
    "learning_start_steps": 1000, # number of env steps to collect before training
    "train_freq": 500, # number of env steps between SAC updates
    "gradient_steps": 200, # number of gradient updates per SAC update
    "batch_size": 1024, # batch size for SAC update
    "eval_freq": 2048, # number of env steps between evaluations
    "replay_size": 1_000_000,
    "gamma": 0.99,
    "tau": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,
    "init_alpha": 0.2,
    "target_entropy": None,
    "save_interval": 50,
}