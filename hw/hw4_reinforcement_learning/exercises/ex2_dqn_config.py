"""
Hyperparameters for Exercise 2 (DQN).

You are encouraged to tune:
- lr
- epsilon
- target_update
- hidden_dim

Please keep the remaining parameters unchanged unless explicitly stated.
"""

DQN_PARAMETERS = {
    # Tuned hyperparameters.
    # lr:            2e-3 learns CartPole within ~200 episodes; 1e-3 is stable
    #                but noticeably slower, 5e-3 starts to diverge.
    # epsilon:       0.01 is enough exploration once the buffer is pre-filled
    #                with `minimal_size` transitions; larger values keep
    #                injecting random actions that cap the achievable return.
    # target_update: 10 updates gives a target network that is fresh enough to
    #                learn quickly while still decoupling the TD target.
    # hidden_dim:    128 units is ample for a 4-dim observation.
    "lr": 2e-3,
    "epsilon": 0.01,
    "target_update": 10,
    "hidden_dim": 128,


    # Fixed parameters
    "gamma": 0.99,
    "num_episodes": 500,
    "buffer_size": 10000,
    "minimal_size": 500,
    "batch_size": 64,
    "seed": 0,
}