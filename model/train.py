from stable_baselines3 import PPO
from utils import make_vec_env, make_env, get_save_name
from model.constants import (
    default_model_params,
    default_env_params,
    logs_path,
    models_path,
)
from datetime import datetime


def train(
    model_params=default_model_params,
    env_params=default_env_params,
    save_name=None,
    logs_path=logs_path,
    models_path=models_path,
    vec_env_num=64,
    verbose=True,
):
    if not save_name:
        save_name = get_save_name(env_params)

    if verbose:
        print("Training model:", save_name)
        print("Model params:", model_params)
        print("Env params:", env_params)

    prev_time = datetime.now()
    env = (
        make_vec_env(env_params=env_params, vec_env_num=vec_env_num)
        if vec_env_num
        else make_env(env_params=env_params)
    )

    model = PPO(
        policy="CNNPolicy",
        env=env,
        tensorboard_log=f"{logs_path}/{save_name}",
        # learning_rate=model_params['learning_rate'],
        learning_rate=(lambda x: x * model_params["learning_rate"]),
        n_steps=model_params["n_steps"],
        ent_coef=model_params["ent_coef"],
        gamma=model_params["gamma"],
        gae_lambda=model_params["gae_lambda"],
        max_grad_norm=model_params["max_grad_norm"],
        vf_coef=model_params["vf_coef"],
        batch_size=model_params["batch_size"],
        n_epochs=model_params["n_epochs"],
        clip_range=model_params["clip_range"],
        clip_range_vf=model_params["clip_range_vf"],
        verbose=verbose
        # device='mps'
    )

    model.learn(total_timesteps=model_params["total_timesteps"])
    model.save(f"{models_path}/{save_name}")
    train_time = datetime.now() - prev_time

    if verbose:
        print("Model saved:", save_name)
        print("Train Time:", train_time)

    return model
