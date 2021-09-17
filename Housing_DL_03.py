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

def plot_result(x, y):
    plt.figure
    plt.plot(x, y)
    # plt.ylim(0, 30)
    plt.show()

for t in range(epochs):
    train_loop(train_dataloader, model, loss_fn, optimizer)
    b = test_loop(test_dataloader, model, loss_fn)
    history.append(b)
    if t % 100 == 0:
        print(f"Epoch {t}")
        print(f"loss: {b}")
print("Done!")

plot_history(history)

val = torch.tensor([[0.80271,0,8.14,0,0.538,5.456,36.6,3.7965,4,307,21,288.99,11.69],
[0.17505,0,5.96,0,0.499,5.966,30.2,3.8473,5,279,19.2,393.43,10.13],
[0.34006,0,21.89,0,0.624,6.458,98.9,2.1185,4,437,21.2,395.04,12.6]])
y2 = torch.tensor([20.2,24.7,19.2])
pred2 = model(val)
print(pred2)
plot_result(pred2, y2)