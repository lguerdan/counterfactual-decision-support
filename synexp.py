import random, torch
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score

from model import *

###########################################################
######## Error model functions
###########################################################

def ccn_model(eta_star, alpha, beta):
    return (1 - beta - alpha)*eta_star + alpha

def eta(x, environment):
    if environment=='sinusoid':
        return .5 + .5 * np.sin(2.9*x + .1)
    elif environment=='low_base_rate_sinusoid':
        return .5-.5 * np.sin(2.9*x+.1)

        # return np.piecewise(x,[
        #     ((-1 <= x) & (x <= .28)),
        #     ((.28< x) & (x <= .71)),
        #     ((0.71 < x) & (x <= 1))],  
        #     [lambda v: .5+.5 *np.sin(.4*v -1.28), 
        #      lambda v: .5 + .5*np.sin(12*v-4.5),
        #      lambda v: .081+1.9*np.power((v-.7), 2) ])
    else: 
        return np.piecewise(x,[
            ((-1 <= x) & (x <= -.5)),
            ((-.5 < x) & (x <= 0.2069)),
            ((0.2069 < x) & (x <= 0.8)),
            ((0.8 < x) & (x <= 1))],  
            [lambda v: -1.5*v-.75, 
             lambda v: 1.4*v+.7,
             lambda v: -1.5*v+1.3,
             lambda v: 1.25*v - .9 ])
    

def pi(x, func):
    if func=='uniform': 
        return np.ones(x.shape)
        
    elif func=='linear': 
        return .2 * x + .4

def generate_syn_data(
    NS,
    y0_pdf='low_base_rate_sinusoid',
    y1_pdf='sinusoid',
    pi_pdf='linear',
    error_min=0.05,
    error_max=0.25
):
    
    alpha_0, alpha_1, beta_0, beta_1 = np.random.uniform(error_min, error_max, 4)

    # Define class probability functions
    x = np.linspace(-1, 1, num=NS)
    eta_star_0 = eta(x, environment=y0_pdf)
    eta_star_1 = eta(x, environment=y1_pdf)
    eta_y_0 = ccn_model(eta_star_0, alpha_0, beta_0)
    eta_y_1 = ccn_model(eta_star_1, alpha_1, beta_1)

    # Sample from target potential outcome class probability distributions
    YS_0 = np.random.binomial(1, eta_star_0, size=NS)
    YS_1 = np.random.binomial(1, eta_star_1, size=NS)

    # Apply measurement error model
    Y_0 = YS_0.copy()
    Y_0[(YS_0==0) & (np.random.binomial(1, alpha_0, size=NS)==1)] = 1
    Y_0[(YS_0==1) & (np.random.binomial(1, beta_0, size=NS)==1)] = 0

    Y_1 = YS_1.copy()
    Y_1[(YS_1==0) & (np.random.binomial(1, alpha_1, size=NS)==1)] = 1
    Y_1[(YS_1==1) & (np.random.binomial(1, beta_1, size=NS)==1)] = 0

    # Apply consistency assumption to observe potential outcomes
    YS, Y = np.zeros(NS, dtype=np.int64), np.zeros(NS, dtype=np.int64)
    D = np.random.binomial(1, pi(x, func=pi_pdf), size=NS)

    YS[D==0] = YS_0[D==0]
    YS[D==1] = YS_1[D==1]

    Y[D==0] = Y_0[D==0]
    Y[D==1] = Y_1[D==1]

    expdf = pd.DataFrame({
        'X': x,
        'YS_0': YS_0,
        'YS_1': YS_1,
        'Y_0': Y_0,
        'Y_1': Y_1,
        'D': D,
        'YS': YS,
        'Y': Y
    })
    
    error_params = {
        'alpha_0': alpha_0,
        'alpha_1': alpha_1, 
        'beta_0': beta_0, 
        'beta_1': beta_1
    }
    
    return expdf, error_params

def get_loaders(train_df, val_df, do, target):
    
    if target == 'YD':
        train_df = train_df[train_df['D'] == do]
        target = 'Y'
    
    X_train = torch.Tensor(train_df['X'].to_numpy())[:, None]
    Y_train = torch.Tensor(train_df[target].to_numpy())[:, None]
    train_data = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=1)

    X_val = torch.Tensor(val_df['X'].to_numpy())[:, None]
    Y_val = torch.Tensor(val_df[f'YS_{do}'].to_numpy())[:, None]
    val_data = torch.utils.data.TensorDataset(X_val, Y_val)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False, num_workers=1)
    
    return train_loader, val_loader


###########################################################
######## Parameter estimation
###########################################################

def ccpe(expdf, do, n_epochs):

    # Don't need error params for surrogate at this stage
    error_params = {
        'alpha': None,
        'beta': None
    }
    
    expdf = expdf.sample(frac=1).reset_index(drop=True)
    expdf = expdf[expdf['D'] == do]
    split_ix = int(expdf.shape[0]*.7)
    train_df, val_df = expdf.iloc[:split_ix,:], expdf.iloc[split_ix:,:]

    train_loader, val_loader = get_loaders(train_df, val_df, do, target='Y')
    model = MLP()
    losses = train(model, train_loader, error_params=error_params, n_epochs=n_epochs)
    x, y, py_hat = evaluate(model, val_loader)
    
    # Compute error parameters from predicted probabilities
    alpha_hat = py_hat.min()
    beta_hat = 1 - py_hat.max()

    debug_info = {
        'val_x': val_df['X'].to_numpy(),
        'val_py': py_hat
    }
    
    return alpha_hat, beta_hat, debug_info

###########################################################
######## Experiments
###########################################################

def run_param_estimation_exp(expdf, error_params, n_epochs=20):
    alpha_0_hat, beta_0_hat, _ = ccpe(expdf, do=0, n_epochs=20)
    alpha_1_hat, beta_1_hat, _ = ccpe(expdf, do=1, n_epochs=20)
    
    exp_results = error_params.copy()
    exp_results['NS'] = expdf.shape[0]

    exp_results['alpha_0_hat'] = alpha_0_hat
    exp_results['alpha_1_hat'] = alpha_1_hat
    exp_results['beta_0_hat'] = beta_0_hat
    exp_results['beta_1_hat'] = beta_1_hat
    
    exp_results['alpha_0_error'] = np.abs(alpha_0_hat-error_params['alpha_0'])
    exp_results['alpha_1_error'] = np.abs(alpha_1_hat-error_params['alpha_1'])
    exp_results['beta_0_error'] = np.abs(beta_0_hat-error_params['beta_0'])
    exp_results['beta_1_error'] = np.abs(beta_1_hat-error_params['beta_1'])
    
    return exp_results

def run_baseline_comparison_exp(expdf, do, error_params, n_epochs=5, train_ratio=.7):

    expdf = expdf.sample(frac=1).reset_index(drop=True)
    split_ix = int(expdf.shape[0]*train_ratio)
    train_df, val_df = expdf.iloc[:split_ix,:], expdf.iloc[split_ix:,:]
    
    exp_results = {
        'model': [],
        'AU-ROC': []
    }

    surrogate_params = {
        'alpha': error_params[f'alpha_{do}'],
        'beta': error_params[f'beta_{do}']
    }
    
    targets = ['Y', 'YD', f'Y_{do}', f'YS_{do}']
    val_scores = {}
    
    for target in targets:
        
        train_loader, val_loader = get_loaders(train_df, val_df, do, target)
        model = MLP()
        losses = train(model, train_loader, error_params=surrogate_params, n_epochs=n_epochs)
        x, y, py_hat = evaluate(model, val_loader)
        auroc = roc_auc_score(y, py_hat)

        exp_results['model'].append(target)
        exp_results['AU-ROC'].append(auroc)
        
        val_scores[target] = {}
        val_scores[target]['x'] = x
        val_scores[target]['y'] = y
        val_scores[target]['py_hat'] = py_hat

    
    return exp_results, val_scores

