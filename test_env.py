from attrdict import AttrDict
from drivers import *
from data.loaders import *


exp_config = AttrDict({

    # Benchmark configuration
    'benchmark': {
        'name': 'synthetic_1D_sinusoidal',
        'NS': 1000,
        'config': {
            'Y0_PDF': 'piecewise_sinusoid',
            'Y1_PDF': 'low_base_rate_sinusoid',
            'PI_PDF': 'linear'
            }
    },

    'log_dir': 'results',
    
    # Experimental setup 
    'learn_weights': True,
    'learn_parameters': True,
    
    # Hyperparameters
    'n_epochs': 3,
    'n_runs': 1,
    
    'target_POs': [0, 1],
    'policy_gamma': 0,
   
    # Data params
    'train_ratio': .7,
    'split_erm': True,
    'crossfit_erm': True,
    'split_ccpe': True,
    'crossfit_ccpe': True,

    'error_params': [AttrDict({
        'alpha_0': 0.5,
        'alpha_1': 0,
        'beta_0': 0.1,
        'beta_1': 0 
    })],

    'baselines': [
        AttrDict({
            'model': 'OBS',
            'target': 'Y',
            'conditional': False,
            'sl': False,
            'reweight': False
        }), AttrDict({
            'model': 'OBS Oracle',
            'target': 'YS',
            'conditional': False,
            'sl': False,
            'reweight': False
        }), AttrDict({
            'model': 'COM',
            'target': 'Y',
            'conditional': True,
            'sl': False,
            'reweight': False
        }), AttrDict({
            'model': 'COM-SL',
            'target': 'Y',
            'conditional': True,
            'sl': True,
            'reweight': False
        }), AttrDict({
            'model': 'RW',
            'target': 'Y',
            'conditional': True,
            'sl': False,
            'reweight': True
        }), AttrDict({
            'model': 'RW-SL',
            'target': 'Y',
            'conditional': True,
            'sl': True,
            'reweight': True
        }), AttrDict({
            'model': 'Proxy Oracle',
            'target': f'Y',
            'conditional': True,
            'sl': False,
            'reweight': False
        }), AttrDict({
            'model': 'Target Oracle',
            'target': f'YS',
            'conditional': True,
            'sl': False,
            'reweight': False
    })]
})


if __name__ == "__main__":
    po_results, te_results =  run_risk_minimization_exp(exp_config, baselines, error_params, exp_name='baseline_comparison')

