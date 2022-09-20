import torch.optim as optim
from tqdm import tqdm
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(1, 40),
          nn.ReLU(),
          nn.Linear(40, 4),
          nn.ReLU(),
          nn.Linear(4, 1),
          nn.Sigmoid()
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)
  

def train(model, train_loader, loss_func, n_epochs):
    
    opt = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    epoch_loss = []

    for epoch in tqdm(range(0, n_epochs)):
        current_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, targets = data
            opt.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            opt.step()
            current_loss += loss.item()
        
        epoch_loss.append(current_loss)
        current_loss = 0.0
        
    return epoch_loss

def evaluate(model, val_loader):
    labels = []
    preds = []
    
    for i, data in enumerate(val_loader, 0):
        inputs, targets = data
        outputs = model(inputs)
        preds.append(outputs)
        labels.append(targets)

    py_hat = torch.cat(preds, dim=0).detach().numpy()
    y = torch.cat(labels, dim=0).detach().numpy()
    
    return y, py_hat