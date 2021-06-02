from models.autoencoder import *
from models.gan import *
from blocks import train_for_epochs, weight_averaging, log
from fed_loda import FedLODA

train_funcs = {'ae': train_ae, 'gan': train_gan}
valid_funcs = {'ae': validate_ae, 'gan': validate_gan}
predict_funcs = {'ae': predict_ae, 'gan': predict_gan}

def fed_avg_training(models_cfg, federated = True):

    if models_cfg['model']['name'] !="loda":

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
                # print(results_test[m['name']])
            log(models_cfg, _round, results_tr, 'train', models_cfg['val_metric'])
            log(models_cfg, _round, results_test, 'test', models_cfg['val_metric'])

    else:
        raise ValueError("LODA does not saupport fed. aver. yet. ")
    

def fed_loda_training(models_cfg):

    print(f'Federated LODA building...')
    floda = FedLODA(**models_cfg['model'])
    X_list = [np.array(next(iter(models_cfg["models"][i]['loaders']["train"]))["x_data"]) for i in range(models_cfg["nr_models"])]
    y_list = [np.array(next(iter(models_cfg["models"][i]['loaders']["train"]))["y_data"]) for i in range(models_cfg["nr_models"])]

    floda.fit(X_list=X_list , y_list=y_list)

    print(f'Validation {models_cfg["val_metric"]}: ')
    results_tr = {}
    results_test = {}
    X_test = np.array(next(iter(models_cfg["models"][0]['loaders']["test"]))["x_data"])
    y_test = np.array(next(iter(models_cfg["models"][0]['loaders']["test"]))["y_data"])

    for i, m in enumerate(models_cfg['models']):
        val_mets_tr = floda.models[i].valid_metrics(X_list[i], y_list[i])
        val_mets_te = floda.models[i].valid_metrics(X_test, y_test)
        results_tr[m['name']] = val_mets_tr
        results_test[m['name']] = val_mets_te
    for i, m in enumerate(models_cfg['models']):
        val_mets_tr = floda.valid_metrics(X_list[i], y_list[i])
        results_tr[f"{m['name']}_fed"] = val_mets_tr
    val_mets_te = floda.valid_metrics(X_test, y_test)
    for m in models_cfg['models']:
        results_test[f"{m['name']}_fed"] = val_mets_te
    log(models_cfg, 1, results_tr, 'train', models_cfg['val_metric'])
    log(models_cfg, 1, results_test, 'test', models_cfg['val_metric'])


