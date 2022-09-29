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

def get_loss(py_hat, y, alpha_d=None, beta_d=None):
    '''Surrogate loss parameterized by alpha_d, beta_d'''

    if not alpha_d and not beta_d:
        loss = torch.nn.BCELoss()
        return loss(py_hat, y) 

    loss = torch.nn.BCELoss(reduction='none')

    phat_y1 = py_hat[y==1]
    phat_y0 = py_hat[y==0]

    y1_losses = ((1-alpha_d)*loss(phat_y1, torch.ones_like(phat_y1)) -
    beta_d*loss(phat_y1, torch.zeros_like(phat_y1))) / (1-beta_d-alpha_d)

    y0_losses = ((1-beta_d)*loss(phat_y0, torch.zeros_like(phat_y0)) -
    alpha_d*loss(phat_y0, torch.ones_like(phat_y0))) / (1-beta_d-alpha_d)

    return torch.cat([y1_losses, y0_losses]).mean()

# def get_loss(py_hat, y, alpha_d=None, beta_d=None):
    # '''Surrogate loss parameterized by alpha_d, beta_d'''

    # if not alpha_d and not beta_d:
    #     loss = torch.nn.BCELoss()
    #     return loss(py_hat, y) 

    # loss = torch.nn.BCELoss()

    # phat_y1 = py_hat[y==1]
    # phat_y0 = py_hat[y==0]

    # y1_losses = ((1-alpha_d)*loss(phat_y1, torch.ones_like(phat_y1)) -
    # beta_d*loss(phat_y1, torch.zeros_like(phat_y1))) / (1-beta_d-alpha_d)

    # y0_losses = ((1-beta_d)*loss(phat_y0, torch.zeros_like(phat_y0)) -
    # alpha_d*loss(phat_y0, torch.ones_like(phat_y0))) / (1-beta_d-alpha_d)

    # return (y1_losses+y0_losses)/2