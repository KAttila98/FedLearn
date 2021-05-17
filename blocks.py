from utils.models import train_ae
import copy
import torch

def train_for_epochs(_round, epochs, m, data='train'):
    loss_tr = 0
    for epoch in range(epochs):
        loss_tr = train_ae(m, data)
        print(f"Round {_round}.\t"
              f"Epoch {epoch}.\t"
              f"loss_{data}_live={loss_tr:.5f}")

    return loss_tr

def weight_averaging(models):

    with torch.no_grad():
        w_avg = copy.deepcopy(models[0]['net'].state_dict())
        for k in w_avg.keys():

            for i in range(1, len(models)):
                w_avg[k] += models[i]['net'].state_dict()[k] # TODO: mintaszámmal súlyozni
            w_avg[k] = torch.div(w_avg[k], len(models))

        for m in models:
            own_state = m['net'].state_dict()
            for name, param in w_avg.items():

                param = param.data
                own_state[name].copy_(param)