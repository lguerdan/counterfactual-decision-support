from attrdict import AttrDict
import torch

from data.benchmarks import synthetic, ohie, jobs

def get_benchmark(benchmark_config, error_params):

    if 'synthetic' in benchmark_config['name']:
        X, Y = synthetic.generate_syn_data(benchmark_config, error_params)

    elif benchmark_config['name'] == 'ohie':
        X, Y = ohie.generate_ohie_data(benchmark_config['path'], error_params)

    elif benchmark_config['name'] == 'jobs':
        X, Y = jobs.generate_jobs_data(benchmark_config, error_params)
    
    split_ix = int(X.shape[0]*.7)
    return X[:split_ix], X[split_ix:], Y[:split_ix], Y[split_ix:]


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
