from utils.models import validate_ae
from blocks import train_for_epochs, weight_averaging

def fed_avg_training(models_cfg):

    print('Initial training')
    for m in models_cfg['models']:
        train_for_epochs(_round=0, epochs=models_cfg['local_epochs'], m=m)

    for _round in range(1, models_cfg['rounds']):

        print(f'federated averaging')
        weight_averaging(models_cfg['models'])

        print(f'local training')
        for m in models_cfg['models']:
            train_for_epochs(_round=_round, epochs=models_cfg['local_epochs'], m=m)

        print(f'round: {_round} -- validation {models_cfg["val_metric"]}: ')
        for m in models_cfg['models']:
            res = validate_ae(model_cfg=m, data='val', threshold=models_cfg['model']['anomaly_trhold'])[models_cfg['val_metric']]
            print(res)
