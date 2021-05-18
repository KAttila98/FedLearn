from models.autoencoder import *
from blocks import train_for_epochs, weight_averaging, log

train_funcs = {'ae': train_ae}
valid_funcs = {'ae': validate_ae}
predict_funcs = {'ae': predict_ae}

def fed_avg_training(models_cfg):

    train_func = train_funcs[models_cfg['model']['name']]
    valid_func = valid_funcs[models_cfg['model']['name']]

    print('Initial training')
    for m in models_cfg['models']:
        train_for_epochs(_round=0, epochs=models_cfg['local_epochs'], m=m, train_func=train_func)

    for _round in range(1, models_cfg['rounds']):

        print(f'federated averaging')
        weight_averaging(models_cfg['models'])

        print(f'local training')
        for m in models_cfg['models']:
            train_for_epochs(_round=_round, epochs=models_cfg['local_epochs'], m=m, train_func=train_func)

        print(f'Round: {_round} -- validation {models_cfg["val_metric"]}: ')
        results_tr = {}
        results_val = {}
        for m in models_cfg['models']:
            results_tr[m['name']] = valid_func(model_cfg=m, data='train', kwargs=models_cfg['model'])
            results_val[m['name']] = valid_func(model_cfg=m, data='val', kwargs=models_cfg['model'])
        log(models_cfg, _round, results_tr, 'train', models_cfg['val_metric'])
        log(models_cfg, _round, results_val, 'val', models_cfg['val_metric'])