
import copy
import torch

def train_for_epochs(_round, epochs, m, train_func,  data='train'):
    loss_tr = 0
    for epoch in range(epochs):
        loss_tr = train_func(m, data)
        print(f"Model {m['name']}.\t"
              f"Round {_round}.\t"
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

def log(models_cfg, _round, results, data, metric):
    models = models_cfg['models']
    writers = models_cfg['writers']
    for m in models:
        res = results[m['name']][metric]
        writer = writers[m['name']]
        print(f"{m['name']}\t"
              f"Round {_round}.\t"
              f"{metric}_{data}={res:.5f}\t")
        writer.add_scalar(f'{metric}/{data}', res, _round)
