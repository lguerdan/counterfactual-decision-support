import matplotlib.pyplot as plt
import seaborn as sns

from synexp import *

def viz_ccpe_estimates(expdf, debug_info, error_params, do, y0_pdf, y1_pdf, pi_pdf):

    X = expdf['X']
    pdf = y0_pdf if do==0 else y1_pdf
    eta_d_star = eta(X, pdf)
    eta_d = ccn_model(eta_d_star, error_params[f'alpha_{do}'], error_params[f'beta_{do}'])
    pix = pi(X, pi_pdf)

    f, axs = plt.subplots(1, 1, sharey=False, figsize=(7,5))

    # Class probability functions
    axs.plot(X, eta_d_star, label=f'$\eta^*_{do}(x)$')
    axs.plot(X, eta_d, label=f'$\eta_{do}(x)$', color='black')
    axs.plot(X, pix, label='$\pi(x)$', color='orange', linestyle='--')
    plt.scatter(debug_info['val_x'], debug_info['val_py'], marker='x', color='black', label=f'$\hat\eta_{do}(x)$')
    plt.legend(bbox_to_anchor=(1.2, 1))
    plt.title(f'risk under D={do}')