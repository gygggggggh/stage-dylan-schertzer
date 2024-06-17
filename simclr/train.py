# train.py

import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import NPYDataset
from training_module import SimCLRModule
import torch
import logging

logging.basicConfig(filename="output.log", level=logging.INFO)


x_train = np.load(
    "/home/sacha/Desktop/stage-dylan-schertzer/stage_dylan/visulisation/npy/x_train.npy"
)
y_train = np.load(
    "/home/sacha/Desktop/stage-dylan-schertzer/stage_dylan/visulisation/npy/y_train.npy"
)

x_test = np.load(
        "/home/sacha/Desktop/stage-dylan-schertzer/stage_dylan/visulisation/npy/x_test.npy"
)
y_test = np.load(
    "/home/sacha/Desktop/stage-dylan-schertzer/stage_dylan/visulisation/npy/y_test.npy"
)

# Create the SimCLR model
model = SimCLRModule()
torch.cuda.empty_cache()
train_dataset = NPYDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=19)
test_dataset = NPYDataset(x_test, y_test)
test_loader = DataLoader(
    test_dataset, batch_size=512, shuffle=False, num_workers=19, drop_last=False
)
trainer = pl.Trainer(max_epochs=100, log_every_n_steps=1)
trainer.fit(model, train_loader, test_loader)
model.eval()
torch.save(model.state_dict(), "simCLR.pth")
