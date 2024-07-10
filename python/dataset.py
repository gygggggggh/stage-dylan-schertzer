# %%
import numpy as np
import torch
from torch.utils.data import Dataset


class NPYDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray) -> None:
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.argsort(torch.rand(self.data.shape[1]))[:2]
        x_filtered = self.data[index, indices]

        noise = torch.randn_like(x_filtered) * 0.1
        x_noisy = x_filtered + noise

        return [e for e in x_noisy]

    def __len__(self):
        return len(self.data)


class NPYDatasetAll(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray) -> None:
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.data = self.data.reshape(-1, 60, 12)
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)
