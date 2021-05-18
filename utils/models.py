import torch
from models.autoencoder import AutoEncoder
from torch.optim import Adam, SGD

optimizers = {
    'Adam': Adam,
    'SGD': SGD
}

losses = {
    'L1Loss': torch.nn.L1Loss,
    'BCE': torch.nn.BCELoss
}

models = {'ae': AutoEncoder}

def build_models(models_cfg):

    model_cls = models.get(models_cfg['model']['name'])

    if not model_cls:
        raise KeyError("No appropriate model found for this config")

    for i, m in enumerate(models_cfg['models']):
        m['name'] = f'm{i}'
        if 'cuda' in models_cfg['dev']:
            m['dev'] = models_cfg['dev'] if torch.cuda.is_available() else 'cpu'

        m['net'] = model_cls(models_cfg['model']).to(m['dev'])

        if 'optimizer' in models_cfg['model']:

            m['net'].apply(m['net'].init_weights)
            m['optimizer'] = optimizers[models_cfg['model']['optimizer']](m['net'].parameters(), lr=models_cfg['model']['lr'])
            m['loss'] = losses[models_cfg['model']['loss']](reduction='none')

    return models_cfg

