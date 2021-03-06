import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from utils.config import parse_args, parse_models_cfg
from utils.data import data_loaders
from utils.models import build_models
from fed_algos import fed_avg_training, fed_loda_training
from models.gan import Gan
# import torch


def run():
    args = parse_args()
    models_cfg = parse_models_cfg(args.models_cfg)

    models_cfg = data_loaders(models_cfg=models_cfg)
    models_cfg = build_models(models_cfg)

    name = f"{os.path.splitext(os.path.basename(args.models_cfg))[0]}"
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs(f'{args.log_dir}/{name}/{time}', exist_ok=True)
    models_cfg['writers'] = {m['name']: SummaryWriter(f'{args.log_dir}/{name}/{time}/{m["name"]}')
                             for m in models_cfg['models']}
    
    if models_cfg['model']['name'] !="loda":             
        fed_avg_training(models_cfg)
    else:
        fed_loda_training(models_cfg)

if __name__ == '__main__':
    run()
    # args = parse_args()
    # models_cfg = parse_models_cfg(args.models_cfg)
    # models_cfg = data_loaders(models_cfg=models_cfg)
    # gan1 = Gan(models_cfg['model'])
    # gan2 = Gan(models_cfg['model'])
    # gan1.apply(gan1.init_weights)
    # gan2.load_state_dict(gan1.state_dict())
    # print(gan1.state_dict()['discriminator.net.0.bias'])
    # print(gan2.state_dict()['discriminator.net.0.bias'])
    #print(gan2.state_dict())
    #print(torch.all(torch.eq(gan1.state_dict()['discriminator.net.0.bias'], gan2.state_dict()['discriminator.net.0.bias'])))

