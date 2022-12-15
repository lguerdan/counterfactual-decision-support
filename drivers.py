import random, torch
import numpy as np
import pandas as pd
import numpy.matlib
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score


from model import *
from benchmarks import synthetic, ohie


def compute_treatment_effects(treatement_effect_preds):
    
    # TODO: Need to add additional logging to also handle the in-sample data
    # ate_hat_is = (treatement_effect_preds[1]['in-sample']['y_hat'] -
    #               treatement_effect_preds[0]['in-sample']['y_hat']).mean()

    # ate_is = (treatement_effect_preds[1]['in-sample']['y'] -
    #           treatement_effect_preds[0]['in-sample']['y']).mean()

    # abs_error_is = ate_hat_is-ate_is


    # Run inference using y_hat computed under both d:=1 and d:=0 for all units in the validation fold
    ate_hat_os = (treatement_effect_preds[1]['out-sample']['y_hat'] -
                         treatement_effect_preds[0]['out-sample']['y_hat']).mean()
    
    # "Ground truth" treatment effect. Compute by taking mean outcome under d=1 - d=0 (should match EDA notebook)
    D = treatement_effect_preds[1]['out-sample']['d']
    Y = treatement_effect_preds[1]['out-sample']['y']
    ate_os = Y[D==1].mean() - Y[D==0].mean()

    abs_error_os = ate_hat_os-ate_os
    
    treatment_effect_metrics = {
        'ate_hat_os': ate_hat_os,
        'ate_os': ate_os,
        'abs_error_os': abs_error_os
        # 'ate_is': ate_is,
        # 'ate_hat_is': ate_hat_is,
        # 'abs_error_is': abs_error_is,
    }
    
    return treatment_effect_metrics

def load_dataset(benchmark_config, error_params):

    if 'synthetic' in benchmark_config['name']:
        X, Y = synthetic.generate_syn_data(benchmark_config, error_params)

    elif benchmark_config['name'] == 'ohie':
        X, Y = ohie.generate_ohie_data(benchmark_config['path'], error_params)

    return X, Y


def get_loaders(X, Y, target, do, conditional, split_frac=.7):

    split_ix = int(X.shape[0]*split_frac)
    
    X_train = X[:split_ix].to_numpy()
    Y_train = Y[:split_ix][target].to_numpy()[:, None]
    pD_train = Y[:split_ix]['pD'].to_numpy()[:, None]
    D_train = Y[:split_ix]['D'].to_numpy()[:, None]

    if conditional:
        X_train = X_train[Y[:split_ix]['D'] == do]
        Y_train = Y_train[Y[:split_ix]['D'] == do]
        pD_train = pD_train[Y[:split_ix]['D'] == do]
        D_train = D_train[Y[:split_ix]['D'] == do]

    X_val = X[split_ix:].to_numpy()
    Y_val = Y[split_ix:][f'YS_{do}'].to_numpy()[:, None]
    pD_val = Y[split_ix:][f'pD'].to_numpy()[:, None]
    D_val = Y[split_ix:][f'D'].to_numpy()[:, None]
    
    train_data = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train), torch.Tensor(pD_train), torch.Tensor(D_train))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=1)
    val_data = torch.utils.data.TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val), torch.Tensor(pD_val), torch.Tensor(D_val))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False, num_workers=1)
    
    return train_loader, val_loader

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


def run_baseline(X, Y, baseline, do, loss_config, n_epochs=5, train_ratio=.7):

    target = baseline['target']
    if 'Oracle' in baseline['model']:
        target += f'_{do}'
    
    conditional = 'OBS' not in baseline['model']

    train_loader, val_loader = get_loaders(X, Y, target, do, conditional)
    model = MLP(n_feats=X.shape[1])
    losses = train(model, target, train_loader, loss_config=loss_config, n_epochs=n_epochs)
    metrics = []
    conditional_outcome_preds = {}

    for (sample, loader) in zip(['in-sample', 'out-sample'], [train_loader, val_loader]):

        x, y, py_hat, d = evaluate(model, loader)
        y_hat = np.zeros_like(y)
        y_hat[py_hat > .5] = 1
        auroc = roc_auc_score(y, py_hat)
        acc = (y_hat == y).mean()

        metrics.append({
            'AU-ROC': auroc,
            'ACC': acc,
            'sample': sample
        })

        conditional_outcome_preds[sample] = {}
        conditional_outcome_preds[sample]['y_hat'] = py_hat
        conditional_outcome_preds[sample]['y'] = y
        conditional_outcome_preds[sample]['d'] = d

    return metrics, conditional_outcome_preds

def run_baseline_comparison_exp(benchmark_config, baselines, param_configs, N_RUNS, n_epochs=5):

    po_results = []
    ate_results = []

    for error_parameters in param_configs:

        for RUN in range(N_RUNS):

            X, Y = load_dataset(benchmark_config, error_parameters)

            for baseline in baselines:
                te_preds = {}

                for do in [0, 1]:

                    print('======================================================================')
                    print(f"RUN: {RUN}, model: {baseline['model']}, alpha: {error_parameters[f'alpha_{do}']}, beta: {error_parameters[f'beta_{do}']}")
                    print('====================================================================== \n')

                    loss_config = {
                        'alpha': error_parameters[f'alpha_{do}'] if 'SL' in baseline['model'] else None,
                        'beta': error_parameters[f'beta_{do}'] if 'SL' in baseline['model'] else None,
                        'do': do,
                        'pd': Y['D'].mean(),
                        'reweight': True if 'RW' in baseline['model'] else False
                    }

                    log_metadata = {
                        'do': do,
                        'model': baseline['model'],
                        'alpha_0': error_parameters[f'alpha_0'],
                        'alpha_1': error_parameters[f'alpha_1'],
                        'beta_0': error_parameters[f'beta_0'],
                        'beta_1': error_parameters[f'beta_1']
                    }

                    # Here: need to store the results of each run and compute the resulting ATE (and policy risk)
                    run_metrics, conditional_outcome_preds = run_baseline(X, Y, baseline, do, loss_config, n_epochs=n_epochs, train_ratio=.7)
                    te_preds[do] = conditional_outcome_preds
                    
                    # Potential outcome metrics
                    for result in run_metrics: 
                        po_results.append({**log_metadata, **result})
                
                ate_result = compute_treatment_effects(te_preds)
                ate_results.append({**log_metadata, **ate_result})

    return pd.DataFrame(po_results), pd.DataFrame(ate_results)

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
