from shuttle.data import load_shuttle, ShuttleDataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

datasets = {'shuttle': ShuttleDataset}
loader_func = {'shuttle': load_shuttle}

def data_loaders(models_cfg, valid_split = 0.1, test_split = 0.1, random_state = 42):

    print("Loading local datasets..")
    load_func = loader_func.get(models_cfg['dataset'])
    dset = datasets.get(models_cfg['dataset'])

    if not load_func:
        raise KeyError("No loader funtion for dataset")

    if not dset:
        raise KeyError("No dataset class found")

    X, Y = load_func()

    x_shuffled, y_shuffled = shuffle(X, Y, random_state=random_state)
    x_tv, x_test, y_tv, y_test = train_test_split(
        x_shuffled, y_shuffled, test_size=test_split, random_state=random_state)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_tv, y_tv, test_size=valid_split, random_state=random_state)

    models_cfg['models'] = [{}]*models_cfg['nr_models']
    data_len = x_train.shape[0] // models_cfg['nr_models']

    for i in range(models_cfg['nr_models']):
        models_cfg['models'][i]['loaders'] = {}

        m_data_x = x_train[i*data_len : (i + 1)*data_len]
        m_data_y = y_train[i*data_len : (i + 1)*data_len]

        dataset = datasets[models_cfg['dataset']](x=m_data_x, y=m_data_y)
        loader = DataLoader(dataset, batch_size=models_cfg['batch_size'], pin_memory=True)
        models_cfg['models'][i]['loaders']['train'] = loader

        models_cfg['model']['input_size'] = dataset.input_size
        models_cfg['model']['output_size'] = dataset.output_size

        val_dataset = datasets[models_cfg['dataset']](x=x_valid, y=y_valid)
        val_loader = DataLoader(val_dataset, batch_size=models_cfg['batch_size'], pin_memory=True)
        models_cfg['models'][i]['loaders']['val'] = val_loader

        test_dataset = datasets[models_cfg['dataset']](x=x_test, y=y_test)
        test_loader = DataLoader(test_dataset, batch_size=models_cfg['batch_size'], pin_memory=True)
        models_cfg['models'][i]['loaders']['test'] = test_loader

    return models_cfg

