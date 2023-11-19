from stable_baselines3 import PPO
from utils import get_save_name
from model.constants import (
    default_model_params,
    default_env_params,
    logs_path,
    models_path,
)
from model.train import train
from model.test import test


def evaluate(
    model_params=default_model_params,
    env_params=default_env_params,
    save_name=None,
    logs_path=logs_path,
    models_path=models_path,
    test_timesteps=100,
    vec_env_num=64,
    verbose=True,
):
    if not save_name:
        save_name = get_save_name(env_params)

    if verbose:
        print("Evaluating model:", save_name)

    model = train(
        model_params=model_params,
        env_params=env_params,
        save_name=save_name,
        logs_path=logs_path,
        models_path=models_path,
        vec_env_num=vec_env_num,
        verbose=verbose,
    )
    mean_reward = test(
        env_params=env_params,
        save_name=save_name,
        model=model,
        steps=test_timesteps,
        verbose=verbose,
    )

    return mean_reward
