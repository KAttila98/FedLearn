import torch
from models.autoencoder import AutoEncoder
from models.loda_dumb import LODA
from torch.optim import Adam, SGD

optimizers = {
    'Adam': Adam,
    'SGD': SGD
}

losses = {
    'L1Loss': torch.nn.L1Loss,
    'MSELoss': torch.nn.MSELoss,
    'BCE': torch.nn.BCELoss,
    'cross_entropy': F.binary_cross_entropy
}

models = {'ae': AutoEncoder, "loda": LODA} # Itt a GAN esetében kell egy a generátornak meg egy a discriminatornak?

def build_models(models_cfg):

    model_cls = models.get(models_cfg['model']['name'])
    # print(model_cls)

    if not model_cls:
        raise KeyError("No appropriate model found for this config")


    for i, m in enumerate(models_cfg['models']):
        m['name'] = f'{models_cfg["model"]["name"]}_{i}'
        if models_cfg['model']['name'] != "loda":
            if 'cuda' in models_cfg['dev']:
                m['dev'] = models_cfg['dev'] if torch.cuda.is_available() else 'cpu'

            m['net'] = model_cls(models_cfg['model']).to(m['dev'])

            if 'optimizer' in models_cfg['model']:
                if i == 0:
                    m['net'].apply(m['net'].init_weights)
                else:
                    m['net'].load_state_dict(models_cfg['models'][0]['net'].state_dict())
                m['optimizer'] = optimizers[models_cfg['model']['optimizer']](m['net'].parameters(), lr=models_cfg['model']['lr'])
                m['loss'] = losses[models_cfg['model']['loss']](reduction='none')

                #print(models_cfg['models'][i]['net'].state_dict()['net.enc_1.0.weight'])

    return models_cfg

