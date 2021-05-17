import torch
from torch import nn

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
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.01)