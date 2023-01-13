from attrdict import AttrDict

from data_loaders import loaders
from model import *


###########################################################
######## Parameter estimation
###########################################################

def learn_parameters(ccpe_dataset, config, true_params):

    if not config.learn_parameters == True:
        return true_params.copy()

    error_params_hat = AttrDict({})

    for do in config.target_POs:
        error_params_hat[f'alpha_{do}'],  error_params_hat[f'beta_{do}'] = crossfit_ccpe(ccpe_dataset, do, config)
    
    return error_params_hat


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

    train_loader, test_loader = loaders.get_loaders(
        X_train=dataset.X_train,
        YCF_train=dataset.Y_train,
        X_test=dataset.X_test,
        YCF_test=dataset.Y_test,
        target='Y', 
        do=do, 
        conditional=True
    )

    # Fit Y ~ X|T=t
    eta = MLP(n_feats=dataset.X_train.shape[1])
    losses = train(eta, train_loader, loss_config=loss_config, n_epochs=config.n_epochs, desc=f"CCPE: {do}")
    _, py_hat = evaluate(eta, test_loader)
    
    # Compute error parameters from predicted probabilities
    alpha_hat = py_hat.min()
    beta_hat = 1 - py_hat.max()

    return py_hat, alpha_hat, beta_hat
