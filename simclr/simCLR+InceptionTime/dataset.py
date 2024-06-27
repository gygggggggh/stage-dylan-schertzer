# %%
import torch
from torch.utils.data import Dataset
import numpy as np

class NPYDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray) -> None:
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rand1 = torch.randint(0, self.data.shape[1], (1,)).item()
        rand2 = torch.randint(0, self.data.shape[1], (1,)).item()
        if rand1 == rand2:
            rand2 = (rand2 + 1) % self.data.shape[1]
        # print(f"rand1: {rand1}, rand2: {rand2}")
        return self.data[index, rand1], self.data[index, rand2]


    def __len__(self):
        return len(self.data)

class NPYDatasetAll(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray) -> None:
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)