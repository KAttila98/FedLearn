from scipy.io import loadmat
from torch.utils.data import Dataset
import numpy as np

def load_shuttle():
    real_mat = loadmat('shuttle/shuttle.mat')

    inputs = real_mat['X']
    labels = real_mat['y']

    return inputs, labels

class ShuttleDataset(Dataset):
    def __init__(self, x, y):

        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

        assert self.x.shape[0] == self.y.shape[
            0], f"Input has {self.x.shape[0]} rows, output has {self.y.shape[0]} rows."

    def __len__(self):
        return (self.x.shape[0])

    @property
    def input_size(self):
        return self.x.shape[1]

    @property
    def output_size(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        xi = self.x[idx]
        yi = self.y[idx]

        return {
            "x_data": xi,
            "y_data": yi,
        }