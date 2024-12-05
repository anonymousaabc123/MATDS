import numpy as np
import pandas as pd
from scipy.special import expit, logit, log_expit

result_dict = {}
n = 15000
p_1 = 14
p_2 = 6
z = np.random.binomial(2, 0.5, size=n).astype(np.double)
x_1 = np.random.normal(z, 5 * z + 3 * (1 - z) + 1 * (2 - z), size=(p_1, n)).T
x_2 = np.random.choice(5, size=(p_2, n)).T
# w = np.random.choice(3, size=10, p=[0.5, 0.25, 0.25])  # w和z没有关系
e = 0.6 * z + 0.2 * (1 - z)
w = np.random.binomial(3, e)  # w和z有关系
tau = (x_1[:, 0] + x_2[:, 1]) / 2
x_f = expit(3 * (z + 2 * (2 * w - 2))+ log_expit(w) + log_expit(z) + z**2 + w**2 + (w - 0.5) * tau)   #重要代码expit\log_expit
#for i in range(x_1.shape[1]):
#     x_f = x_f * x_1[:, i] + log_expit(x_1[:, i])
# for j in range(x_2.shape[1]):
#     x_f -= x_2[:, j] + (x_2[:, j] ** 2)
# for j in range(x_2.shape[1]):
#      x_f += x_1[:, j] * x_2[:, j]
b = expit(x_f)  
y = np.random.binomial(1, b)

for i in range(x_1.shape[1]):
    name = 'x' + str(i)
    result_dict[name] = x_1[:,i]
for i in range(x_2.shape[1]):
    name = 'xx' + str(i)
    result_dict[name] = x_2[:, i]
result_dict['treatment'] = w
result_dict['label'] = y
res_df = pd.DataFrame(result_dict)
# print(res_df)
res_df.to_csv("~/Downloads/Four_treatment_synthetic_new_data_ver2.csv")

w_num_dict = {}
for i in range(w.shape[0]):
    t = w[i]
    if t not in w_num_dict:
        w_num_dict[t] = [0, 0]
    w_num_dict[t][0] += y[i]
    w_num_dict[t][1] += 1
print(w_num_dict)


