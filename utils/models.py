import torch
from models.autoencoder import AutoEncoder
from torch.optim import Adam, SGD
import numpy as np
from sklearn.metrics import roc_auc_score

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

    for m in models_cfg['models']:
        if 'cuda' in models_cfg['dev']:
            m['dev'] = models_cfg['dev'] if torch.cuda.is_available() else 'cpu'

        m['net'] = model_cls(models_cfg['model']).to(m['dev'])

        if 'optimizer' in models_cfg['model']:

            m['net'].apply(m['net'].init_weights)
            m['optimizer'] = optimizers[models_cfg['model']['optimizer']](m['net'].parameters(), lr=models_cfg['model']['lr'])
            m['loss'] = losses[models_cfg['model']['loss']](reduction = 'none')

    return models_cfg

def train_ae(model_cfg, data):

    net = model_cfg['net']
    loader_tr = model_cfg['loaders'][data]
    optimizer = model_cfg['optimizer']
    dev = model_cfg['dev']
    loss = model_cfg['loss']

    net.train()
    loss_sum = 0.0
    loss_count = 0
    for b in loader_tr:
        optimizer.zero_grad()

        X = b["x_data"].to(dev)

        xhat = net(X)

        output = loss(xhat, X).sum()

        output_n = output / X.shape[0]

        output_n.backward()

        optimizer.step()

        loss_sum += output.detach() / X.shape[0]
        loss_count += 1

    loss_tr = loss_sum / loss_count

    return loss_tr

def validate_ae(model_cfg, data, threshold = 0.9):
    net = model_cfg['net']
    loss = model_cfg['loss']

    loader = model_cfg['loaders'][data]
    dev = model_cfg['dev']
    net.eval()

    y_true_list = []
    y_hat_list = []
    pred_list = []

    with torch.no_grad():
        for b in loader:
            X = b["x_data"].to(dev)
            y_data = b["y_data"].view(-1).to(dev)
            y_true_list.append(y_data)
            x_hat = net(X)
            output = loss(x_hat, X).mean(dim=1).view(-1)
            y_hat_list.append(output)
            output_sorted = output.sort().values
            th_value = output_sorted[int(output.shape[0] * threshold) - 1].item()
            pred_list.append(torch.where(output >= th_value, 1.0, 0.0))

        y_true = torch.cat(y_true_list, dim=0).cpu().numpy()
        y_hat = torch.cat(y_hat_list, dim=0).cpu().numpy()
        preds = torch.cat(pred_list, dim=0).cpu().numpy()

        eq = y_true == preds
        accuracy = np.mean(eq)

        auc = roc_auc_score(y_true, y_hat)

        return {
            'accuracy': accuracy,
            'auc': auc,
        }

def predict_ae(model_cfg, data, threshold = 0.9):
    net = model_cfg['net']
    loader = model_cfg['loaders'][data]
    dev = model_cfg['dev']
    loss = model_cfg['loss']
    net.eval()

    preds = []
    with torch.no_grad():
        for b in loader:
            X = b["x_data"].to(dev)
            x_hat = net(X)
            output = loss(x_hat, X).mean(dim = 1).view(-1)
            output_sorted = output.sort().values
            th_value = output_sorted[int(output.shape[0] * threshold) - 1].item()
            preds.append(torch.where(output >= th_value, 1.0, 0.0))

        preds = torch.cat(preds, dim=0).cpu().numpy()

    return preds
