import pandas as pd
import numpy as np
from scipy.special import expit, logit, log_expit

res_dict = {}
n=15000
p=15

z = np.random.binomial(1, 0.5, size=n).astype(np.double)
X = np.random.normal(z, 5 * z + 3 * (1 - z), size=(p, n)).T
e = 0.6 * z + 0.4 * (1 - z)
w = np.random.binomial(3, e)
x_f = expit(3 * (z + 2 * (2 * w - 2)))    # 重要代码expit\log_expit
b = expit(x_f)
y = np.random.binomial(1, b)
# res_dict['user_id'] = [i for i in range(n)]
for i in range(X.shape[1]):
    name = 'x' + str(i)
    res_dict[name] = X[:,i]
res_dict['treatment'] = w
res_dict['label'] = y
res_df = pd.DataFrame(res_dict)
# print(res_df)
res_df.to_csv("~/Downloads/Four_treatment_synthetic_old_data_ver2.csv")

w_num_dict = {}
for i in range(w.shape[0]):
    t = w[i]
    if t not in w_num_dict:
        w_num_dict[t] = [0, 0]
    w_num_dict[t][0] += y[i]
    w_num_dict[t][1] += 1
print(w_num_dict)

