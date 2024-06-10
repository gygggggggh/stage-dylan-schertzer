import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from lightly.models import SimCLR
from pytorch_lightning import LightningModule, Trainer
from torch.optim import Adam
import numpy as np
from sklearn.metrics import accuracy_score

# Custom dataset for numpy arrays
class NPYDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).long()

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

# LightningModule for SimCLR
class SimCLRModule(LightningModule):
    def __init__(self):
        super().__init__()
        base_model = resnet50(weights=None)
        base_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)        
        self.model = SimCLR(backbone=base_model, num_ftrs=1000)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        print(f"Original input shape: {x.shape}")
        x = x.view(x.size(0), 1, 268, 268)  
        print(f"Reshaped input shape: {x.shape}")
        return self.model(x)
 
    def training_step(self, batch, batch_idx):
        x, y = batch
        z0 = self.forward(x)
        z1 = self.forward(y)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        logits = torch.argmax(z, dim=1)
        accuracy = accuracy_score(y.cpu(), logits.cpu())
        self.log('test_accuracy', accuracy)
        return {'test_accuracy': accuracy} 

def main():
    # Load your numpy arrays
    x_train = np.load("stage_dylan/visulisation/npy/x_train.npy")
    y_train = np.load("stage_dylan/visulisation/npy/y_train.npy")
    x_test = np.load("stage_dylan/visulisation/npy/x_test.npy")
    y_test = np.load("stage_dylan/visulisation/npy/y_test.npy")

    # Create the datasets
    train_dataset = NPYDataset(x_train, y_train)
    test_dataset = NPYDataset(x_test, y_test)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=19)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=19)

    # Create the SimCLR model
    model = SimCLRModule()

    # Train the model
    trainer = Trainer(max_epochs=50, accumulate_grad_batches=4)
    trainer.fit(model, train_loader, test_loader)

    # Save the model
    torch.save(model.state_dict(), "model.pth")

    # Load the model
    model = SimCLRModule()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    # Test the model
    test_accuracy = trainer.test(model, test_loader)
    print(f"Test accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
