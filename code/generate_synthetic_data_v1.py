import pandas as pd
import numpy as np
from scipy.special import expit, logit

res_dict = {}
n=10000
p=15
z = np.random.binomial(1, 0.5, size=n).astype(np.double)
X = np.random.normal(z, 5 * z + 3 * (1 - z), size=(p, n)).T
e = 0.75 * z + 0.25 * (1 - z)
w = np.random.binomial(2, e)
b = expit(3 * (z + 2 * (2 * w - 2)))
y = np.random.binomial(1, b)
#res_dict['user_id'] = [i for i in range(n)]
for i in range(X.shape[1]):
    name = 'x' + str(i)
    res_dict[name] = X[:,i]
res_dict['treatment'] = w
res_dict['label'] = y
res_df = pd.DataFrame(res_dict)
print(res_df)
res_df.to_csv("~/Downloads/three_treatment_synthetic_bias_data.csv")
