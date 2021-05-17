from utils.config import parse_args, parse_models_cfg
from utils.data import data_loaders
from utils.models import build_models
from fed_algos import fed_avg_training

def run():
    args = parse_args()
    models_cfg = parse_models_cfg(args.models_cfg)

    models_cfg = data_loaders(models_cfg=models_cfg)
    models_cfg = build_models(models_cfg)

    fed_avg_training(models_cfg)

if __name__ == '__main__':
    run()

