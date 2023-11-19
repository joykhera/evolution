from stable_baselines3 import PPO
from utils import make_vec_env, make_env, get_save_name
from model.constants import default_env_params, models_path
from datetime import datetime
import matplotlib.pyplot as plt


def test(
    env_params=default_env_params,
    steps=100,
    save_name=None,
    model=None,
    verbose=True,
):
    if not save_name:
        save_name = get_save_name(env_params)

    if not model:
        model = PPO.load(f"{models_path}/{save_name}")

    env = make_env(env_params=env_params, vec_env=False)
    obs, info = env.reset(test=True)
    rewards = []

    if verbose:
        print("Testing model:", save_name)

    prev_time = datetime.now()

    for i in range(steps):
        action, _states = model.predict([obs])
        # action, _states = model.predict([[obs]], deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if type(reward) == list:
            reward = reward[0]
        else:
            reward = reward
        rewards.append(reward)

        if verbose:
            print(
                f"Step: {i:2}, Info: {info}, Prediction: {action[0]}, Reward: {reward}"
            )

        if done:
            env.reset(test=True)

    mean_reward = sum(rewards) / len(rewards)
    test_time = datetime.now() - prev_time

    if verbose:
        print("Mean Reward:", mean_reward)
        print("Min Reward:", min(rewards))
        print("Max Reward:", max(rewards))
        # print('Max possible reward:', env.unwrapped.max_possible_reward())
        print("Test Time:", test_time)
        print("steps", steps)

        env.unwrapped.render_all()
        plt.show()

    return mean_reward
