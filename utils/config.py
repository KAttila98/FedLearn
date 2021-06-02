import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Federated anomaly detection")
    parser.add_argument("--models_cfg", help="Models config file", type=str, default="configs\\fed_gan.yml")
    parser.add_argument("--log_dir", help="Logging directory", type=str, default="logs")

    return parser.parse_args()

def parse_models_cfg(config_file):
    with open(config_file, 'r') as f:
        models_cfg = yaml.safe_load(f)
    return models_cfg