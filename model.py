import torch

import torch.optim as optim
from tqdm import tqdm
from torch import nn

class MLP(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(n_feats, 40),
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

def get_prop_weights(x, pd, loss_config):

    if not loss_config['reweight']:
        return None

    do = loss_config['do']
    weights = pd/(((2*do-1)*pd) + (1-do))

    return weights

def train(model, target, train_loader, loss_config, n_epochs):
    
    opt = optim.Adam(model.parameters(), lr=.001)
    epoch_loss = []

    for epoch in tqdm(range(0, n_epochs), desc=f"Target: {target}"):
        current_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            x, y, pd, _ = data
            balancing_weights = get_prop_weights(x, pd, loss_config)
            opt.zero_grad()
            outputs = model(x)
            loss = get_loss(outputs, y, loss_config, weights=balancing_weights)
            loss.backward()
            opt.step()
            current_loss += loss.item()
        
        epoch_loss.append(current_loss)
        current_loss = 0.0
        
    return epoch_loss

def evaluate(model, data_loader):

    labels = []
    preds = []
    feats = []
    treatments = []
    
    for i, data in enumerate(data_loader, 0):
        x, y, _, d = data
        outputs = model(x)
        preds.append(outputs)
        labels.append(y)
        feats.append(x)
        treatments.append(d)

    x = torch.cat(feats, dim=0).detach().numpy()
    y = torch.cat(labels, dim=0).detach().numpy()
    py_hat = torch.cat(preds, dim=0).detach().numpy().squeeze()
    d = torch.cat(treatments, dim=0).detach().numpy().squeeze()

    return x, y, py_hat, d

def get_loss(py_hat, y, loss_config, weights):
    '''Surrogate loss parameterized by alpha, beta'''

    loss = torch.nn.BCELoss(reduction='none')
    alpha, beta = loss_config['alpha'], loss_config['beta']
    if alpha != None and beta != None:
        phat_y1 = py_hat[y==1]
        phat_y0 = py_hat[y==0]

        y1_losses = ((1-alpha)*loss(phat_y1, torch.ones_like(phat_y1)) -
        beta*loss(phat_y1, torch.zeros_like(phat_y1))) / (1-beta-alpha)

        y0_losses = ((1-beta)*loss(phat_y0, torch.zeros_like(phat_y0)) -
        alpha*loss(phat_y0, torch.ones_like(phat_y0))) / (1-beta-alpha)
        val_loss = torch.cat([y1_losses, y0_losses])

    else:
        val_loss = loss(py_hat, y)

    if weights != None:
        val_loss = val_loss*weights

    return val_loss.mean()

