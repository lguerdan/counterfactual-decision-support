import random, torch
import numpy as np
import pandas as pd
import numpy.matlib
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score

from model import *
from benchmarks import synthetic, ohie, jobs


###########################################################
######## Utils and data loaders
###########################################################

def compute_treatment_metrics(po_preds, Y_test, benchmark, policy_gamma=0):

    D, pD, E, YS_0, YS_1, YS = Y_test['D'],  Y_test['pD'], Y_test['E'], Y_test['YS_0'], Y_test['YS_1'], Y_test['YS']
    YS_0_hat = po_preds[0]
    YS_1_hat = po_preds[1]

    # Below uses hard-coded values. Can also compute over sample via:
    # ate = YS_1[(D==1) & (E==1)].mean() - YS_0[(D==0) & (E==1)].mean()

    if 'synthetic' in benchmark:
        ate = YS_1.mean() - YS_0.mean()

    elif benchmark == 'ohie':
        ate = -0.00340

    elif benchmark == 'jobs':
        ate = -0.07794

    else: 
        raise Exception("Invalid benchmark")

    # Evaluate over factual and counterfactual outcomes
    # E=1 is required for experimental sub-sample of NSW study
    ate_hat = YS_1_hat[(E==1)].mean() - YS_0_hat[(E==1)].mean()

    # Simulate treatment policy
    pi = np.zeros_like(D)
    pi[YS_1_hat-YS_0_hat > policy_gamma] = 1

    # Compute propensities via ''ground truth'' treatment probabilities
    inv_weights = pD.copy()
    inv_weights[D==0] = 1-pD
    inv_weights = 1 - inv_weights

    # Compute policy risk
    policy_risk_num = (YS * (pi == D) * inv_weights).sum()
    policy_risk_demon = (pi == D).sum()

    treatment_effect_metrics = {
        'ate': ate,
        'ate_hat': ate_hat,
        'ate_error': abs(ate-ate_hat),
        'policy_risk': policy_risk_num/policy_risk_demon
    }
    
    return treatment_effect_metrics

def load_dataset(benchmark_config, error_params):
    # This is a good place to experimentally manipulate selection bias
    # Based on a configurable parameter (or algorithmic)

    if 'synthetic' in benchmark_config['name']:
        X, Y = synthetic.generate_syn_data(benchmark_config, error_params)

    elif benchmark_config['name'] == 'ohie':
        X, Y = ohie.generate_ohie_data(benchmark_config['path'], error_params)

    elif benchmark_config['name'] == 'jobs':
        X, Y = jobs.generate_jobs_data(benchmark_config, error_params)

    return X, Y


def get_loaders(X_train, YCF_train, X_test, YCF_test, target, do, conditional):

    if conditional:
        X_train = X_train[YCF_train['D']==do]
        YCF_train = YCF_train[YCF_train['D']==do]
    
    X_train = X_train.to_numpy()
    Y_train = YCF_train[target].to_numpy()[:, None]
    pD_train = YCF_train['pD'].to_numpy()[:, None]
    D_train = YCF_train['D'].to_numpy()[:, None]

    X_test = X_test.to_numpy()
    Y_test = YCF_test[f'YS_{do}'].to_numpy()[:, None]
    pD_test = YCF_test['pD'].to_numpy()[:, None]
    D_test = YCF_test['D'].to_numpy()[:, None]
    
    train_data = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train), torch.Tensor(pD_train), torch.Tensor(D_train))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=1)
    test_data = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test), torch.Tensor(pD_test), torch.Tensor(D_test))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=1)
    
    return train_loader, test_loader

###########################################################
######## Risk minimmization experiments
###########################################################

def run_risk_minimization_exp(exp_config, baselines, param_configs, N_RUNS, n_epochs=5, train_ratio=.7):

    po_results = []
    te_results = []

    for error_params in param_configs:

        for RUN in range(N_RUNS):


            #TODO: update dataset loading here to 
            # - (1) insert selection bias and 
            # - (2) estimate weights based on a configurable parameter
            X, Y = load_dataset(exp_config['benchmark'], error_params)
            split_ix = int(X.shape[0]*train_ratio)

            dataset = {
                'X_train': X[:split_ix],
                'Y_train': Y[:split_ix],
                'X_test': X[split_ix:],
                'Y_test': Y[split_ix:]
            }

            for baseline in baselines:

                print('===============================================================================================================')
                print(f"RUN: {RUN}, model: {baseline['model']}, alpha_0: {error_params['alpha_0']}, alpha_1: {error_params['alpha_1']}, beta_0: {error_params['beta_0']}, beta_1: {error_params['beta_1']}")
                print('=============================================================================================================== \n')

                loss_config = {
                    'pd': Y['D'].mean(),
                    'reweight': True if 'RW' in baseline['model'] else False
                }

                po_metrics, te_metrics = run_baseline_two_sided(
                    dataset=dataset,
                    baseline=baseline,
                    error_params=error_params,
                    loss_config=loss_config,
                    exp_config=exp_config,
                    n_epochs=n_epochs)
                
                po_results.extend(po_metrics)
                te_results.append(te_metrics)

    return pd.DataFrame(po_results), pd.DataFrame(te_results)


def run_baseline_two_sided(dataset, baseline, error_params,
        loss_config, exp_config, n_epochs=5):

    log_metadata = error_params.copy()
    log_metadata['model'] = baseline['model']
    po_metrics = []
    po_preds = {}

    for do in [0, 1]:

        # Setup invervention-specific configuration parameters
        loss_config['alpha'] = error_params[f'alpha_{do}'] if 'SL' in baseline['model'] else None
        loss_config['beta'] = error_params[f'beta_{do}'] if 'SL' in baseline['model'] else None
        loss_config['do'] = do

        do_metrics, y_hat = run_baseline_one_sided(dataset, baseline, do, loss_config, n_epochs)
        po_metrics.append({**log_metadata, **do_metrics, 'do': do})
        po_preds[do] = y_hat
    
    treatment_effects = compute_treatment_metrics(po_preds, dataset['Y_test'], exp_config['benchmark']['name'], exp_config['policy_gamma'])
    te_metrics = {**log_metadata, **treatment_effects}

    return po_metrics, te_metrics


def run_baseline_one_sided(dataset, baseline, do, loss_config, n_epochs=5):

    target, baseline_name = baseline['target'], baseline['model']
    if 'Oracle' in baseline_name:
        target += f'_{do}'
    
    conditional = 'OBS' not in baseline_name

    train_loader, test_loader = get_loaders(
        X_train=dataset['X_train'],
        YCF_train=dataset['Y_train'],
        X_test=dataset['X_test'],
        YCF_test=dataset['Y_test'],
        target=target, 
        do=do, 
        conditional=conditional
    )
        
    model = MLP(n_feats=dataset['X_train'].shape[1])
    losses = train(model, target, train_loader, loss_config=loss_config, n_epochs=n_epochs)
    metrics, py_hat = evaluate(model, test_loader)

    return metrics, py_hat
  
###########################################################
######## Parameter estimation
###########################################################

def ccpe(env, X, Y, do, target, n_epochs):

    loss_config = {
        'alpha': None,
        'beta':  None,
        'prop_func': pi,
        'pi_pdf': env['PI_PDF'],
        'do': do,
        'reweight': False
    }
    
    train_loader, val_loader = get_loaders(X, Y, target, do, conditional=True)
    model = MLP(n_feats=X.shape[1])
    losses = train(model, 'Y|D', train_loader, loss_config=loss_config, n_epochs=n_epochs)
    x, y, py_hat = evaluate(model, val_loader)
    
    # Compute error parameters from predicted probabilities
    alpha_hat = py_hat.min()
    beta_hat = 1 - py_hat.max()

    return alpha_hat, beta_hat

def run_estimation_error_exp(do, param_configs, error_param, NS, N_RUNS, n_epochs=5, train_ratio=.7):
    
    baseline = {
        'model': 'Conditional outcome (SL)',
        'target': 'Y0'
    }

    results = []
    for config in param_configs:
        surrogate = {}

        if error_param == 'alpha':
            alpha = config['param']
            surrogate['alpha'] = config['estimate']
            beta = 0
            surrogate['beta'] = 0

        if error_param == 'beta':
            beta = config['param']
            surrogate['beta'] = config['estimate']
            alpha = 0
            surrogate['alpha'] = 0
            
        for run in range(N_RUNS):

            X, Y, error_params = generate_syn_data(
                NS=NS,
                K=1,
                y0_pdf=Y0_PDF,
                y1_pdf=Y1_PDF,
                pi_pdf=PI_PDF,
                alpha_min=alpha,
                alpha_max=alpha+.001,
                beta_min=beta,
                beta_max=beta+.001,
                shuffle=True
            )

            run = run_baseline(X, Y, baseline, do=0,
                                    surrogate_params=surrogate, n_epochs=n_epochs, train_ratio=.7)
            result = {
                'AU-ROC': run['AU-ROC'],
                'ACC': run['ACC'],
                'alpha': alpha,
                'beta': beta, 
                'alpha_hat': surrogate['alpha'],
                'beta_hat': surrogate['beta'],
                'error_param': error_param
            }

            results.append(result)
        
    return results

def ccpe_benchmark_exp(env, param_configs, SAMPLE_SIZES, N_RUNS, n_epochs):

    exp_results = {
        'alpha': [],
        'beta': [],
        'alpha_hat': [],
        'beta_hat': [],
        'alpha_error': [],
        'beta_error': [],
        'NS': []
    }

    for config in param_configs:
        for NS in SAMPLE_SIZES:
            print('======================================================================')
            print(f"NS: {SAMPLE_SIZES}, alpha: {config['alpha']}, beta: {config['beta']}")
            print('====================================================================== \n')
            for RUN in range(N_RUNS):

                X, Y, error_params = generate_syn_data(
                    NS=NS,
                    K=1,
                    y0_pdf=env['Y0_PDF'],
                    y1_pdf=env['Y1_PDF'],
                    pi_pdf=env['PI_PDF'],
                    alpha_0=config['alpha'],
                    alpha_1=config['alpha'],
                    beta_0=config['beta'],
                    beta_1=config['beta'],
                )

                alpha_hat, beta_hat = ccpe(env, X, Y, target=f'Y0', do=0, n_epochs=n_epochs)
                
                exp_results['alpha'].append(config['alpha'])
                exp_results['beta'].append(config['beta'])
                exp_results['alpha_hat'].append(alpha_hat)
                exp_results['beta_hat'].append(beta_hat)
                exp_results['alpha_error'].append(alpha_hat - config['alpha'])
                exp_results['beta_error'].append(beta_hat - config['beta'])
                exp_results['NS'].append(NS)

    return exp_results
