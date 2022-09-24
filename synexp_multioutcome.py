import random, torch
import numpy as np
import pandas as pd
import numpy.matlib
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
    K=1,
    y0_pdf='sinusoid',
    y1_pdf='low_base_rate_sinusoid',
    pi_pdf='linear',
    error_min=0.05,
    error_max=0.25
):  

    alpha_0_arr = np.random.uniform(error_min, error_max, K)
    alpha_1_arr = np.random.uniform(error_min, error_max, K)
    beta_0_arr = np.random.uniform(error_min, error_max, K)
    beta_1_arr = np.random.uniform(error_min, error_max, K)

    # Define class probability functions
    x = np.linspace(-1, 1, num=NS)
    eta_star_0 = eta(x, environment=y0_pdf)
    eta_star_1 = eta(x, environment=y1_pdf)

    # Sample from target potential outcome class probability distributions
    YS_0 = np.random.binomial(1, eta_star_0, size=NS)
    YS_1 = np.random.binomial(1, eta_star_1, size=NS)

    Y_0 = np.matlib.repmat(YS_0, K, 1).T
    Y_1 = np.matlib.repmat(YS_1, K, 1).T

    alpha_0_errors = np.array([np.random.binomial(1, alpha_0_arr[i], size=NS) for i in range(K)]).T
    alpha_1_errors = np.array([np.random.binomial(1, alpha_1_arr[i], size=NS) for i in range(K)]).T

    beta_0_errors = np.array([np.random.binomial(1, beta_0_arr[i], size=NS) for i in range(K)]).T
    beta_1_errors = np.array([np.random.binomial(1, beta_1_arr[i], size=NS) for i in range(K)]).T

    Y_0[alpha_0_errors == 1] = 1
    Y_0[beta_0_errors == 1] = 0

    Y_1[alpha_1_errors == 1] = 1
    Y_1[beta_1_errors == 1] = 0

    # Apply consistency assumption to observe potential outcomes
    YS = np.zeros(NS, dtype=np.int64)
    Y = np.zeros_like(Y_0)

    D = np.random.binomial(1, pi(x, func=pi_pdf), size=NS)
    YS[D==0] = YS_0[D==0]
    YS[D==1] = YS_1[D==1]

    Y[D==0,:] = Y_0[D==0,:]
    Y[D==1,:] = Y_1[D==1,:]
        
    dataset = {
        'X': x,
        'YS_0': YS_0,
        'YS_1': YS_1,
        'D': D,
        'YS': YS
    }

    for yx in range(Y.shape[1]):
        dataset[f'Y{yx}'] = Y[:,yx]
        dataset[f'Y{yx}_0'] = Y_0[:,yx]
        dataset[f'Y{yx}_1'] = Y_1[:,yx]

    error_params = {
        'alpha_0': alpha_0_arr,
        'alpha_1': alpha_1_arr,
        'beta_0': beta_0_arr,
        'beta_1': beta_1_arr
    }

    return pd.DataFrame(dataset), error_params

def get_loaders(train_df, val_df, do, target, conditional):
    
    if conditional:
        train_df = train_df[train_df['D'] == do]
    
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

def ccpe(expdf, do, target, n_epochs):

    # Don't need error params for surrogate at this stage
    error_params = {
        'alpha': None,
        'beta': None
    }
    
    split_ix = int(expdf.shape[0]*.7)
    train_df, val_df = expdf.iloc[:split_ix,:], expdf.iloc[split_ix:,:]

    train_loader, val_loader = get_loaders(train_df, val_df, do, target=target, conditional=True)
    model = MLP()
    losses = train(model, 'Y|D', train_loader, error_params=error_params, n_epochs=n_epochs)
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


def run_baseline(expdf, baseline, do, surrogate_params, n_epochs=5, train_ratio=.7):
    
    expdf = expdf.sample(frac=1).reset_index(drop=True)
    split_ix = int(expdf.shape[0]*train_ratio)
    train_df, val_df = expdf.iloc[:split_ix,:], expdf.iloc[split_ix:,:]
    target = baseline['target']

    conditional = True if 'Conditional' in baseline['model'] else False
    
    # Train model
    train_loader, val_loader = get_loaders(train_df, val_df, do, target, conditional)
    model = MLP()
    losses = train(model, target, train_loader, error_params=surrogate_params, n_epochs=n_epochs)
    
    # Evaluate on validation data
    x, y, py_hat = evaluate(model, val_loader)
    y_hat = (py_hat > .5)
    auroc = roc_auc_score(y, py_hat)
    acc = (y_hat == y).mean()

    results = {}
    results['AU-ROC'] = auroc
    results['ACC'] = acc
    results['x'] = x
    results['y'] = y
    results['py_hat'] = py_hat

    return results


def run_baseline_comparison_exp(baselines, do,  N_RUNS, NS, K=1, n_epochs=5):

    Y0_PDF = 'sinusoid'
    Y1_PDF = 'low_base_rate_sinusoid'
    PI_PDF = 'linear'
        
    exp_results = {
        'model': [],
        'AU-ROC': [],
        'ACC': []
    }
    
    for RUN in range(N_RUNS):

        expdf, error_params = generate_syn_data(
            NS,
            K,
            y0_pdf=Y0_PDF,
            y1_pdf=Y1_PDF,
            pi_pdf=PI_PDF,
            error_min=0.05,
            error_max=0.25
        )
        
        for baseline in baselines:
            target = baseline['target']
            surrogate_params = {
                'alpha': error_params[f'alpha_{do}'] if baseline['target'] == 'Conditional outcome (SL)' else None,
                'beta': error_params[f'beta_{do}'] if baseline['target'] == 'Conditional outcome (SL)' else None
            }
            results = run_baseline(expdf, baseline, do, surrogate_params, n_epochs=n_epochs, train_ratio=.7)
            exp_results['model'].append(baseline['model'])
            exp_results['AU-ROC'].append(results['AU-ROC'])
            exp_results['ACC'].append(results['ACC'])
            
    return exp_results

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

