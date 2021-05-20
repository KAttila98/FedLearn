import torch
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score

class AutoEncoder(nn.Module):
    def __init__(self, kwargs):

        super(AutoEncoder, self).__init__()
        self.net = nn.Sequential()

        self.net.add_module('enc_0', nn.Sequential(
            nn.Linear(in_features=kwargs['input_size'], out_features=kwargs['hidden_sizes'][0]),
            nn.ReLU()))

        for i in range(1, kwargs['nr_enc_lyrs']):
            self.net.add_module(f'enc_{i}', nn.Sequential(
                nn.Linear(in_features=kwargs['hidden_sizes'][i-1], out_features=kwargs['hidden_sizes'][i]),
                nn.ReLU()))

        for i in range(0, kwargs['nr_enc_lyrs'] - 1):
            self.net.add_module(f'dec_{i}', nn.Sequential(
                nn.Linear(in_features=kwargs['hidden_sizes'][kwargs['nr_enc_lyrs'] - i - 1], out_features=kwargs['hidden_sizes'][kwargs['nr_enc_lyrs'] - i - 2]),
                nn.ReLU()))

        self.net.add_module(f'dec_{kwargs["nr_enc_lyrs"] - 1}', nn.Sequential(
            nn.Linear(in_features=kwargs['hidden_sizes'][0], out_features=kwargs['input_size']),
            nn.ReLU()))

    def forward(self, x):
        for lyr in self.net:
            x = lyr(x)

        return x

    def init_weights(self, m):
        if type(m) in [nn.Linear]:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu') # ugyanaz az init!!
            m.bias.data.fill_(0.01)


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

def validate_ae(model_cfg, data, kwargs):
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
            th_value = output_sorted[int(output.shape[0] * kwargs['anomaly_trhold']) - 1].item()
            pred_list.append(torch.where(output > th_value, 1.0, 0.0))

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

def predict_ae(model_cfg, data, kwargs):
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
            th_value = output_sorted[int(output.shape[0] * kwargs['anomaly_trhold']) - 1].item()
            preds.append(torch.where(output > th_value, 1.0, 0.0))

        preds = torch.cat(preds, dim=0).cpu().numpy()

    return preds