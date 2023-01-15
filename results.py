import glob
import pandas as pd


def get_erm_convergence_result_df(exp_name):
  results = []
  for f in glob.glob(f'results/{exp_name}/*PO.csv'):
    results.append(pd.read_csv(f))
  nsdf = pd.concat(results)
  return nsdf, nsdf.groupby(['baseline', 'NS']).mean()
