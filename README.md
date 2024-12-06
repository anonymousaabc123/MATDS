The Code and dataset for the paper “MATDS: Uplift Modeling for Multi-Attribute Treatment under Distribution Shift”

## The synthetic dataset
To construct a synthetic dataset with multiple treatments and inherent bias, while enhancing the complexity of the relationship between $X$ and $Y$, two types of covariates are integrated: an adjusted normal distribution and values generated using np.random.choice(). Three implementation methods are given in the code directory
```
- generate_synthetic_data_v1.py: Treatment={0,1,2}, Y={0, 1}, weakly biased
- generate_synthetic_data_v2.py: Treatment={0,1,2,3}, Y={0, 1},  strongly biased
- generate_synthetic_data_v3.py: Treatment={0,1,2,3}, Y={0, 1}, strongly biased
```
Taking the data generated by generate_synthetic_data_v3.py as an example, the data statistics are as follows
| Treatment | Count  | Label_Sum | Response_rate |
|:--------:| :---:|  :---: | :---:  |
| 0 |  2098   | 1031 |  0.491   |
| 1 |  5408   | 3565 |  0.659   |
| 2 |  5453   | 4028 |  0.739   |
| 3 |  2041   | 1476 |  0.723   |

## The public MIMIC-III dataset
We cleaned three multi-treatment uplift datasets from three different dimensions.
- The impact of different hospital services on discharge rate; # The problem of hadm_id duplication needs to be considered, which is caused by the duplication of data by the service;
- The impact of different disease types of treatment on discharge rates; # The duplication problem, because a user may have multiple diseases, needs to be handled
- The impact of different drugs on discharge rate under the same disease;
In our paper, we use the cleaned MIMIC-III data of the second dimension. The data generation code and analysis are in the code directory.

