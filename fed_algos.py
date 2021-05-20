from models.autoencoder import *
from blocks import train_for_epochs, weight_averaging, log

train_funcs = {'ae': train_ae}
valid_funcs = {'ae': validate_ae}
predict_funcs = {'ae': predict_ae}

def fed_avg_training(models_cfg, federated = True):

    train_func = train_funcs[models_cfg['model']['name']]
    valid_func = valid_funcs[models_cfg['model']['name']]

    for _round in range(1, models_cfg['rounds'] + 1):

        print(f'local training')
        for m in models_cfg['models']:
            train_for_epochs(_round=_round, epochs=models_cfg['local_epochs'], m=m, train_func=train_func)

        if federated:
            print(f'federated averaging')
            weight_averaging(models_cfg['models'])

        print(f'Round: {_round} -- validation {models_cfg["val_metric"]}: ')
        results_tr = {}
        results_test = {}
        for m in models_cfg['models']:
            results_tr[m['name']] = valid_func(model_cfg=m, data='train', kwargs=models_cfg['model'])
            results_test[m['name']] = valid_func(model_cfg=m, data='test', kwargs=models_cfg['model'])
            print(results_test[m['name']])
        log(models_cfg, _round, results_tr, 'train', models_cfg['val_metric'])
        log(models_cfg, _round, results_test, 'test', models_cfg['val_metric'])
