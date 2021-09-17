import Housing_Datasete_01 as hd01
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Dataset を作成する。
dataset = hd01.Housing("C:/Users/S2212357/Documents/Z6_DataBase/DeepLeaning/housing.csv")
n_samples = len(dataset)
train_size = int(len(dataset) * 0.8)
test_size = n_samples - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
# DataLoader を作成する。
train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(13, 50),
            nn.RReLU(),
            nn.Linear(50, 10),
            nn.RReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

learning_rate = 1e-3
epochs = 500
history = []

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X).squeeze()
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X).squeeze()
            test_loss += loss_fn(pred, y).item()
    test_loss /= size
    return test_loss

def plot_history(dict):
    plt.figure
    plt.plot(range(1, epochs+1), dict)
    plt.ylim(0, 30)
    plt.show()
    pass

for t in range(epochs):
    train_loop(train_dataloader, model, loss_fn, optimizer)
    b = test_loop(test_dataloader, model, loss_fn)
    history.append(b)
    if t % 100 == 0:
        print(f"Epoch {t}")
        print(f"loss: {b}")
print("Done!")

plot_history(history)
