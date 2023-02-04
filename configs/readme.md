



## Setup
```
! pip3 install -r requirements.txt
```

## Data
For licensing and privacy purposes, we omit data from this package. Data can be downloaded from: 
- JOBS: [train](https://www.fredjo.com/files/jobs_DW_bin.new.10.train.npz), [test](https://www.fredjo.com/files/jobs_DW_bin.new.10.test.npz)
- [OHIE](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SJG1ED): Raw data can be  



## Synthetic experiment

```
! python3 drivers.py erm A_main_experiment_jobs_oracle_R10
```

## Semi-synthetic experiments
```
! python3 drivers.py erm_experimental A_main_experiment_jobs_oracle_R10
! python3 drivers.py erm_experimental A_main_experiment_ohie_oracle_R10
```
