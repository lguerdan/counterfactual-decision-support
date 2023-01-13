import numpy as np
import pandas as pd
import numpy.matlib
from attrdict import AttrDict
import torch

from data_loaders.benchmarks import synthetic, ohie, jobs
import ccpe, erm, model

###########################################################
######## Risk minimmization experiments
###########################################################

def run_risk_minimization_exp(config, baselines, param_configs, exp_name):

    te_results = []
    po_results = []
    
    for error_params in param_configs:
        for run_num in range(config.n_runs):

            print('===============================================================================================================')
            print(f"RUN: {run_num}, alpha_0: {error_params.alpha_0}, alpha_1: {error_params.alpha_1}, beta_0: {error_params.beta_0}, beta_1: {error_params.beta_1}")
            print('=============================================================================================================== \n')

            te_baseline_metrics, po_baseline_metrics = erm.run_model_comparison(config, baselines, error_params)
            te_results.extend(te_baseline_metrics)
            po_results.extend(po_baseline_metrics)

    po_df, te_df = pd.DataFrame(po_results), pd.DataFrame(te_results)
    po_df.to_csv(f'{config.log_dir}/baseline_comparison_runs={config.n_runs}_epochs={config.n_epochs}_benchmark={config.benchmark.name}_samples={config.NS}_PO.csv')
    te_df.to_csv(f'{config.log_dir}/baseline_comparison_runs={config.n_runs}_epochs={config.n_epochs}_benchmark={config.benchmark.name}_samples={config.NS}_TE.csv')

    return po_df, te_df

  
###########################################################
######## Parameter estimation experiments
###########################################################

def run_ccpe_exp(config, error_param_configs, sample_sizes, N_RUNS, do=0, n_epochs=5, train_ratio=.7):
    
    results = []
    for error_params in error_param_configs:
        for NS in sample_sizes:

            config['benchmark']['NS'] = NS

            for RUN in range(N_RUNS):
                X, Y = loader.get_benchmark(config['benchmark'], error_params)
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

    ccpe_results = pd.DataFrame(results)
    ccpe_results.to_csv(f'{config.log_dir}/parameter_estimation_runs={config.n_runs}_epochs={config.n_epochs}_benchmark={config.benchmark.name}.csv')
        
    return ccpe_results
