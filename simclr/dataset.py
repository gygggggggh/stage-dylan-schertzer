# %%
import torch
from torch.utils.data import Dataset


class NPYDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float()

    def __getitem__(self, index):
        rand1 = torch.randint(0, self.data.shape[1], (1,)).item()
        rand2 = torch.randint(0, self.data.shape[1], (1,)).item()
        if rand1 == rand2:
            rand2 = (rand2 + 1) % self.data.shape[1]
        # print(f"rand1: {rand1}, rand2: {rand2}")
        return self.data[index, rand1], self.data[index, rand2]

    @staticmethod
    def getall(x_test, y_test):
        x_test = x_test.reshape(-1, 60, 12)
        return torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()

    def __len__(self):
        return len(self.data)
