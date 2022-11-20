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

def get_prop_weights(x, loss_config):

    if not loss_config['reweight']:
        return None

    prop_func = loss_config['prop_func']
    do = loss_config['do']
    pd = loss_config['pd']
    pi_pdf = loss_config['pi_pdf']

    # Assuming access to ''perfect'' weights
    pdx = prop_func(x, func=pi_pdf)

    weights = pd/(((2*do-1)*pdx) + (1-do))

    return weights

def train(model, target, train_loader, loss_config, n_epochs):
    
    opt = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    epoch_loss = []

    for epoch in tqdm(range(0, n_epochs), desc=f"Target: {target}"):
        current_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            weights = get_prop_weights(x, loss_config)
            opt.zero_grad()
            outputs = model(x)
            loss = get_loss(outputs, y, loss_config, weights)
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

def get_loss(py_hat, y, loss_config, weights):
    '''Surrogate loss parameterized by alpha, beta'''

    alpha_d, beta_d = loss_config['alpha'], loss_config['beta']

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

    # loss = torch.nn.BCELoss(reduction='none')

    # alpha, beta = loss_config['alpha'], loss_config['beta']

    # if alpha and beta:
    #     phat_y1 = py_hat[y==1]
    #     phat_y0 = py_hat[y==0]

    #     y1_losses = ((1-alpha)*loss(phat_y1, torch.ones_like(phat_y1)) -
    #     beta*loss(phat_y1, torch.zeros_like(phat_y1))) / (1-beta-alpha)

    #     y0_losses = ((1-beta)*loss(phat_y0, torch.zeros_like(phat_y0)) -
    #     alpha*loss(phat_y0, torch.ones_like(phat_y0))) / (1-beta-alpha)

    #     val_loss = torch.cat([y1_losses, y0_losses])

    # else:
    #     val_loss = loss(py_hat, y)

    # if weights != None:
    #     print('vall loss corrected')
    #     val_loss = val_loss*weights

    # return val_loss.mean()

