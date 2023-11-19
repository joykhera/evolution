default_model_params = {
    "total_timesteps": 10000000,
    "learning_rate": 0.00025,
    "n_steps": 128,
    "ent_coef": 0.01,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
    "batch_size": 128,
    "n_epochs": 4,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "verbose": 1,
}

default_env_params = env_params = {
    "size": 100,
    "grid_size": 10,
}

logs_path = "training/logs"
models_path = "training/models"
