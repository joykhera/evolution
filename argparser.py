import argparse
import yaml


def get_save_name(defaults, args, short_names):
    diff_items = {}
    for key, value in defaults.items():
        if key in args and args[key] != value:
            diff_items[key] = args[key]
    save_name = ",".join(f"{short_names[key]}={value}" for key, value in diff_items.items())
    return save_name or "default"


def parse_args():
    defaults = yaml.safe_load(open("defaults.yaml"))
    default_args = defaults["env_config"] | defaults["model_config"]
    short_names = defaults['short_names']

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", action="store_true", help="Enable training mode.")
    parser.add_argument("-test", action="store_true", help="Enable testing mode.")
    parser.add_argument("-cp", "--checkpoint", type=str, help="Checkpoint from which to load the model.")
    parser.add_argument("-sn", "--save_name", type=str, default="", help="Custom name for the experiment.")

    for key, value in default_args.items():
        parser.add_argument(f"-{short_names[key]}", f"--{key}", type=type(value), default=value, help=f"{key} parameter for the experiment.")

    args = vars(parser.parse_args())

    if args['test'] and not args["checkpoint"]:
        raise ValueError("Please specify a checkpoint to load the model from with -cp")

    if args["save_name"] == "":
        args["save_name"] = get_save_name(default_args, args, short_names)

    new_model_config = {key: args.pop(key) for key in list(defaults["model_config"].keys())}
    new_env_config = {key: args.pop(key) for key in list(defaults["env_config"].keys())}

    return args, new_model_config, new_env_config
