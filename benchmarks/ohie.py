import pandas as pd
import numpy as np
from numpy.random import permutation


def generate_ohie_data(OHIE_PATH, error_params, shuffle=True):
    
    ohie_df = pd.read_csv(OHIE_PATH)
    
    YS = ohie_df[['Y']].squeeze()
    D = ohie_df[['D']].squeeze()
    ohie_df.drop(columns=['Y', 'D'], inplace=True)
    ohie_df = (ohie_df - ohie_df.mean(axis=0))/ohie_df.std(axis=0)

    YS_0 = np.zeros_like(YS)
    YS_1 = np.zeros_like(YS)
    Y_0 = np.zeros_like(YS)
    Y_1 = np.zeros_like(YS)
    Y = np.zeros_like(YS)

    YS_0[D==0] = YS[D==0]
    YS_1[D==1] = YS[D==1]

    alpha_0_errors = np.random.binomial(1, error_params['alpha_0'], size=ohie_df.shape[0])
    alpha_1_errors = np.random.binomial(1, error_params['alpha_1'], size=ohie_df.shape[0])
    beta_0_errors = np.random.binomial(1, error_params['beta_0'], size=ohie_df.shape[0])
    beta_1_errors = np.random.binomial(1, error_params['beta_1'], size=ohie_df.shape[0])

    Y_0[alpha_0_errors == 1] = 1
    Y_0[beta_0_errors == 1] = 0

    Y_1[alpha_1_errors == 1] = 1
    Y_1[beta_1_errors == 1] = 0

    Y[D==0] = Y_0[D==0]
    Y[D==1] = Y_1[D==1]

    dataset_y = {
        'YS': YS,
        'YS_0': YS_0,
        'YS_1': YS_1,
        'Y_0': Y_0,
        'Y_1': Y_1,
        'Y': Y,
        'pD': np.ones_like(D) * D.mean(),
        'D': D,
        'E': np.ones_like(YS) # Include for computign the ATT on JOBS test data
    }

    X, Y = pd.DataFrame(ohie_df), pd.DataFrame(dataset_y)

    if shuffle: 
        suffle_ix = permutation(X.index)
        X = X.iloc[suffle_ix]
        Y = Y.iloc[suffle_ix]

    return X, Y