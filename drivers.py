import random, torch
import numpy as np
import pandas as pd
import numpy.matlib
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score
from attrdict import AttrDict

from model import *
from benchmarks import synthetic, ohie, jobs


###########################################################
######## Utils and data loaders
###########################################################

def compute_crossfit_metrics(crossfit_erm_preds, Y_test, n_splits, config, log_metadata):
    
    te_metrics = []
    po_metrics = []
    
    for baseline_name, results in crossfit_erm_preds.items():

        po_preds = {}
        for do in exp_config.target_POs:

            y = Y_test[f'YS_{do}']
            po_preds[do] = np.zeros_like(y)
            
            # Compute aggregate model prediction on evaluation fold
            for split in range(n_splits): 
                po_preds[do] = np.add(po_preds[do], (1/n_splits)*results[split][do])

                po_result = {
                    'AU-ROC': roc_auc_score(y, po_preds[do]),
                    'ACC': (po_preds[do] == y).mean(),
                    'do': do,
                    'baseline': baseline_name
                }

            po_metrics.append({**log_metadata, **po_result})

        if len(exp_config.target_POs) == 2:
            te_result = compute_treatment_metrics(po_preds, Y_test, benchmark, policy_gamma=0)
            te_metrics.append({**log_metadata, **te_result, 'baseline': baseline_name })
            
    return te_metrics, po_metrics


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



############# TODO in this function ##############
## Move code below to data_loaders.py
##################################################

def get_splits(X_train, X_test, Y_train, Y_test, config):
    
    N_train = X_train.shape[0]
    
    if not config.split_erm:
        dataset = AttrDict({
            'X_train': X_train,
            'X_test': X_test,
            'Y_train': Y_train,
            'Y_test': Y_test
        })
        return [dataset.deepcopy(), dataset.deepcopy(), dataset.deepcopy()]
    
    else:
        split_ix_1, split_ix_2 = int(.33*N_train), int(.66*N_train)

        split1 = AttrDict({
            'X_train': X_train.iloc[:split_ix_1, :],
            'X_test': X_test,
            'Y_train': Y_train.iloc[:split_ix_1, :],
            'Y_test': Y_test
        })

        split2 = AttrDict({
            'X_train': X_train.iloc[split_ix_1:split_ix_2, :],
            'X_test': X_test,
            'Y_train': Y_train.iloc[split_ix_1:split_ix_2, :],
            'Y_test': Y_test
        })

        split3 = AttrDict({
            'X_train': X_train.iloc[split_ix_2:, :],
            'X_test': X_test,
            'Y_train': Y_train.iloc[split_ix_2:, :],
            'Y_test': Y_test
        })
        
        return [split1, split2, split3]

def load_benchmark(benchmark_config, error_params):

    if 'synthetic' in benchmark_config['name']:
        X, Y = synthetic.generate_syn_data(benchmark_config, error_params)

    elif benchmark_config['name'] == 'ohie':
        X, Y = ohie.generate_ohie_data(benchmark_config['path'], error_params)

    elif benchmark_config['name'] == 'jobs':
        X, Y = jobs.generate_jobs_data(benchmark_config, error_params)
    
    split_ix = int(X.shape[0]*.7)
    return X[:split_ix], X[split_ix:], Y[:split_ix], Y[split_ix:]


def get_loaders(X_train, YCF_train, X_test, YCF_test, target, do, conditional):

    if conditional:
        X_train = X_train[YCF_train['D']==do]
        YCF_train = YCF_train[YCF_train['D']==do]

    eval_target = 'D' if target == 'D' else f'YS_{do}'
    
    X_train = X_train.to_numpy()
    Y_train = YCF_train[target].to_numpy()[:, None]
    pD_train = YCF_train['pD'].to_numpy()[:, None]
    D_train = YCF_train['D'].to_numpy()[:, None]

    X_test = X_test.to_numpy()
    Y_test = YCF_test[eval_target].to_numpy()[:, None]
    pD_test = YCF_test['pD'].to_numpy()[:, None]
    D_test = YCF_test['D'].to_numpy()[:, None]
    
    train_data = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train), torch.Tensor(pD_train), torch.Tensor(D_train))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=1)
    test_data = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test), torch.Tensor(pD_test), torch.Tensor(D_test))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=1)
    
    return train_loader, test_loader


def learn_parameters(ccpe_dataset, config, true_params):

    if not config.learn_parameters == True:
        return true_params.copy()

    error_params_hat = AttrDict({})

    for do in config.target_POs:
        error_params_hat[f'alpha_{do}'],  error_params_hat[f'beta_{do}'] = crossfit_ccpe(ccpe_dataset, do, config)
    
    return error_params_hat
    

def learn_weights(weight_dataset, config):
    '''
        Estimate weighting function on training dataset and run inference on evaluation fold
    '''

    train_loader, test_loader = get_loaders(
        X_train=weight_dataset.X_train,
        YCF_train=weight_dataset.Y_train,
        X_test=weight_dataset.X_test,
        YCF_test=weight_dataset.Y_test,
        target='D', 
        do=0, 
        conditional=False
    )

    loss_config = AttrDict({
        'pd': weight_dataset.Y_train['D'].mean(),
        'reweight': False,
        'alpha': None,
        'beta': None
    })
        
    model = MLP(n_feats=weight_dataset.X_train.shape[1])
    losses = train(model, train_loader, loss_config=loss_config, n_epochs=config.n_epochs, desc='Propensity model')

    return model

###########################################################
######## Risk minimmization experiments
###########################################################


def run_risk_minimization_exp(config, baselines, param_configs):

    te_results = []
    po_results = []
    
    for error_params in param_configs:
        for run_num in range(config.n_runs):

            print(error_params)

            print('===============================================================================================================')
            print(f"RUN: {run_num}, alpha_0: {error_params.alpha_0}, alpha_1: {error_params.alpha_1}, beta_0: {error_params.beta_0}, beta_1: {error_params.beta_1}")
            print('=============================================================================================================== \n')

            te_baseline_metrics, po_baseline_metrics = run_baselines(config, baselines, error_params)
            te_results.extend(te_baseline_metrics)
            po_results.append(po_baseline_metrics)

    return pd.DataFrame(po_results), pd.DataFrame(te_results)


def run_baselines(config, baselines, error_params):

    # TODO: load_benchmark should insert environment-specific selection bias only to X_train/Y_train
    X_train, X_test, Y_train, Y_test = load_benchmark(config.benchmark, error_params)

    crossfit_erm_preds = { baseline.model: {} for baseline in baselines }

    if config.crossfit_erm == True:
        split_permuations = [(0,1,2), (0,2,1), (2,0,1)]
    else:
        split_permuations = [(0,1,2)]
    
    for s_ix, (p,q,r) in enumerate(split_permuations):

        data_splits = get_splits(X_train, X_test, Y_train, Y_test, config)
        weight_dataset, ccpe_dataset, erm_dataset = data_splits[p], data_splits[q], data_splits[r]

        if config.learn_weights:
            propensity_model = learn_weights(weight_dataset, config)

        error_params_hat = learn_parameters(ccpe_dataset, config, true_params=error_params)

        for baseline in baselines:

            loss_config = AttrDict({
                'd_mean': Y_test['D'].mean(),
                'reweight': baseline.reweight,
            })
            baseline.propensity_model = propensity_model if config.learn_weights else None
            baseline.error_params_hat = error_params_hat

            crossfit_erm_preds[baseline.model][s_ix] = run_erm_split(
                erm_dataset=erm_dataset,
                baseline_config=baseline,
                loss_config=loss_config,
                exp_config=config
            )
    log_metadata = AttrDict(error_params)
    log_metadata.benchmark = config.benchmark.name
    te_metrics, po_metrics = compute_crossfit_metrics(crossfit_erm_preds, Y_test, len(split_permuations), config, log_metadata)


def run_erm_split(erm_dataset, baseline_config, loss_config, exp_config):
    '''
        erm_split: [erm train split (.33 of training data), ERM test split (test data)]    
    '''

    po_preds = {}

    for do in exp_config.target_POs:

        loss_config.alpha = baseline_config.error_params_hat[f'alpha_{do}'] if baseline_config.sl else None
        loss_config.beta = baseline_config.error_params_hat[f'beta_{do}'] if baseline_config.sl else None
        loss_config.do = do

        train_loader, test_loader = get_loaders(
            X_train=erm_dataset.X_train,
            YCF_train=erm_dataset.Y_train,
            X_test=erm_dataset.X_test,
            YCF_test=erm_dataset.Y_test,
            target=baseline_config.target, 
            do=do,
            conditional=baseline_config.conditional
        )

        model = MLP(n_feats=erm_dataset.X_train.shape[1])
        propensity_model = baseline_config.propensity_model if exp_config.learn_weights else None
        losses = train(model, train_loader, loss_config=loss_config, n_epochs=exp_config.n_epochs, desc=f"ERM: {baseline_config.model}")
        _, py_hat = evaluate(model, test_loader)        
        po_preds[do] = py_hat
    
    return po_preds

  
###########################################################
######## Parameter estimation
###########################################################

def crossfit_ccpe(ccpe_dataset, do, config):

    X_train, Y_train = ccpe_dataset['X_train'], ccpe_dataset['Y_train']
    split_ix = int(X_train.shape[0]*.5)

    ccpe_split_1 = AttrDict({
        'X_train': X_train.iloc[split_ix:, :],
        'Y_train': Y_train.iloc[split_ix:, :],
        'X_test': X_train.iloc[:split_ix, :],
        'Y_test': Y_train.iloc[:split_ix, :],
    })

    ccpe_split_2 = AttrDict({
        'X_train': X_train.iloc[:split_ix, :],
        'Y_train': Y_train.iloc[:split_ix, :],
        'X_test': X_train.iloc[split_ix:, :],
        'Y_test': Y_train.iloc[split_ix:, :],
    })

    _, alpha_1_hat, beta_1_hat = ccpe(ccpe_split_1, do, config)
    _, alpha_2_hat, beta_2_hat = ccpe(ccpe_split_2, do, config)

    return (alpha_1_hat+alpha_2_hat)/2, (beta_1_hat+beta_2_hat)/2

def ccpe(dataset, do, config):
    '''
        Fit class probability function and evaluate min/max on held-out data
    '''

    loss_config = AttrDict({
        'alpha': None,
        'beta':  None,
        'do': do,
        'reweight': False
    })

    train_loader, test_loader = get_loaders(
        X_train=dataset.X_train,
        YCF_train=dataset.Y_train,
        X_test=dataset.X_test,
        YCF_test=dataset.Y_test,
        target='Y', 
        do=do, 
        conditional=True
    )

    # Fit Y ~ X|T=t
    model = MLP(n_feats=dataset.X_train.shape[1])
    losses = train(model, train_loader, loss_config=loss_config, n_epochs=config.n_epochs, desc=f"CCPE: {do}")
    _, py_hat = evaluate(model, test_loader)
    
    # Compute error parameters from predicted probabilities
    alpha_hat = py_hat.min()
    beta_hat = 1 - py_hat.max()

    return py_hat, alpha_hat, beta_hat

def run_ccpe_exp(config, error_param_configs, sample_sizes, N_RUNS, do=0, n_epochs=5, train_ratio=.7):
    
    results = []
    for error_params in error_param_configs:
        for NS in sample_sizes:

            config['benchmark']['NS'] = NS

            for RUN in range(N_RUNS):
                X, Y = load_benchmark(config['benchmark'], error_params)
                split_ix = int(X.shape[0]*config['train_test_ratio'],)

                dataset = {
                    'X_train': X[:split_ix],
                    'Y_train': Y[:split_ix],
                    'X_test': X[split_ix:],
                    'Y_test': Y[split_ix:]
                }

                alpha_hat, beta_hat = ccpe(dataset, do, config)

                results.append({
                    'NS': NS,
                    'alpha': error_params[f'alpha_{do}'],
                    'beta': error_params[f'beta_{do}'],
                    'alpha_hat': alpha_hat,
                    'beta_hat': beta_hat,
                    'alpha_error': error_params[f'alpha_{do}'] - alpha_hat,
                    'beta_error': error_params[f'beta_{do}'] - beta_hat
                })
        
    return pd.DataFrame(results)
