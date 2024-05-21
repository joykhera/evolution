import argparse
import yaml

def parse_args():

    defaults = yaml.safe_load(open("defaults.yaml"))
    args = defaults["env_config"] | defaults["model_config"]
    short_names = defaults['short_names']

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", action="store_true", help="Enable training mode.")
    parser.add_argument("-test", action="store_true", help="Enable testing mode.")
    parser.add_argument("-cp", "--checkpoint", type=str, help="Checkpoint from which to load the model.")
    parser.add_argument("-sn", "--save_name", type=str, default="my_experiment", help="Custom name for the experiment.")

    for key, value in args.items():
        parser.add_argument(f"-{short_names[key]}", f"--{key}", type=type(value), default=value, help=f"{key} parameter for the experiment.")

    args = vars(parser.parse_args())

    new_model_config = {key: args.pop(key) for key in list(defaults["model_config"].keys())}
    new_env_config = {key: args.pop(key) for key in list(defaults["env_config"].keys())}

    return args, new_model_config, new_env_config
