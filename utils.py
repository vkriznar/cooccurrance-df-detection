from PIL import Image
from torch.utils import data
import torch
import numpy as np

class Dataset_Csv_CO(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, folders, labels, transform=None):
        "Initialization"
        # self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_tensors(self, path, use_transform):
        with open(path, 'rb') as f:
            tensor = np.load(f)
        f.close()
        """ if use_transform is not None:
            tensor = use_transform(tensor) """
        return tensor

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_tensors(folder, self.transform)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y
