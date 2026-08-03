PPO_PARAMETERS = {
    "seed": 42,
    "hidden_sizes": [256, 128, 128],
    "total_iterations": 500, # total number of training iterations
    "n_steps": 2048, # number of env steps before each update
    "mini_batch_size": 1024, # batch size for PPO update
    "n_epochs": 10, # number of epochs per PPO update
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "surrogate_loss_coeff": 1.0,
    "value_loss_coeff": 0.01,
    "entropy_coeff": 0.005,
    "clip_ratio": 0.2,
    "learning_rate": 3e-4,
    "target_kl": 0.01,
    "max_grad_norm": 0.5,
    "save_interval": 50,
}