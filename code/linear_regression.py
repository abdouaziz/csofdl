import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np

from tqdm import tqdm  # for progress bar

from sklearn.datasets import load_wine


config = {
    "input_size": 11,
    "output_size": 1,
    "batch_size": 16,
    "shuffle": True,
    "num_workers": 2,
    "learning_rate": 0.001,
    "epochs": 100,
}


class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt("./data/wine.csv", delimiter="," , dtype=np.float32 , skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])  # n_samples, 1
        self.y = [torch.tensor([0] if i[0] < 5 else [1]) for i in self.y]
        self.y = torch.stack(self.y)

        self.n_samples = xy.shape[0]

    def __len__(self):
        # len(dataset)
        return self.n_samples

    def __getitem__(self, index):
        # dataset[0]
        return {"x": self.x[index].float(), "y": self.y[index].float()}


def data_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def train_step(model, data_loader, optimizer, criterion):
    model.train()
    for batch in tqdm(data_loader):
    
        x = batch["x"]
        y = batch["y"]

        # forward
        y_pred = model(x)
        loss = criterion(y_pred, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def test_step(model, data_loader, criterion):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            x = batch["x"]
            y = batch["y"]
            # forward
            y_pred = model(x)
            loss = criterion(y_pred, y)

    return loss.item()


def main():
    # data
    dataset = WineDataset()
    train_loader = data_loader(dataset, config["batch_size"], config["shuffle"])
    test_loader = data_loader(dataset, config["batch_size"], False)

    # model
    model = LinearRegression(config["input_size"], config["output_size"])

    # loss
    criterion = nn.MSELoss()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

    # train
    for epoch in range(config["epochs"]):
        train_loss = train_step(model, train_loader, optimizer, criterion)
        test_loss = test_step(model, test_loader, criterion)
        print(f"Epoch : {epoch+1} , Train Loss : {train_loss} , Test Loss : {test_loss}")

    # save model
    torch.save(model.state_dict(), "./model/linear_regression.pth")


if __name__ == "__main__":
    main()
