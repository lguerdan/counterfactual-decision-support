import torch

import torch.optim as optim
from tqdm import tqdm
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(1, 40),
          nn.ReLU(),
          nn.Linear(40, 20),
          nn.ReLU(),
          nn.Linear(20, 4),
          nn.ReLU(),
          nn.Linear(4, 1),
          nn.Sigmoid()
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)
  

def train(model, target, train_loader, error_params, n_epochs):
    
    opt = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    epoch_loss = []

    for epoch in tqdm(range(0, n_epochs), desc=f"Target: {target}"):
        current_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            opt.zero_grad()
            outputs = model(x)
            loss = get_loss(outputs, y, error_params['alpha'], error_params['beta'])
            loss.backward()
            opt.step()
            current_loss += loss.item()
        
        epoch_loss.append(current_loss)
        current_loss = 0.0
        
    return epoch_loss

def evaluate(model, val_loader):

    labels = []
    preds = []
    feats = []
    
    for i, data in enumerate(val_loader, 0):
        inputs, targets = data
        outputs = model(inputs)
        preds.append(outputs)
        labels.append(targets)
        feats.append(inputs)

    x = torch.cat(feats, dim=0).detach().numpy()
    y = torch.cat(labels, dim=0).detach().numpy()
    py_hat = torch.cat(preds, dim=0).detach().numpy()
    
    return x, y, py_hat

def get_loss(y_hat, y, alpha_d=None, beta_d=None):

    loss_func = torch.nn.BCELoss()
    if not alpha_d:
        return loss_func(y_hat, y) 
        
    pos_loss = loss_func(y_hat[y==1], y[y==1])
    neg_loss = loss_func(y_hat[y==0], y[y==0])

    pos_label_losses = ((1-alpha_d)*pos_loss - beta_d*neg_loss)/(1-beta_d-alpha_d)
    neg_label_losses = ((1-beta_d)*neg_loss - alpha_d*pos_loss)/(1-beta_d-alpha_d)

    return pos_label_losses + neg_label_losses
