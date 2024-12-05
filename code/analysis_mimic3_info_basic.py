import pandas as pd
pd.set_option('display.max_columns', 999)
import pandas.io.sql as psql
# plot a figure directly on Notebook
import matplotlib.pyplot as plt

# ~/Downloads/mimic-iii-clinical-database-demo-1.4
# /Desktop/data/uplift_related/mimic-iii-clinical-database-1.4
a = pd.read_csv("/Desktop/data/uplift_related/mimic-iii-clinical-database-1.4/ADMISSIONS.csv")
a.columns = map(str.lower, a.columns)
# 数据分析
# a.groupby(['marital_status']).count()['row_id'].plot(kind='pie')
# a.groupby(['religion']).count()['row_id'].plot(kind = 'barh')
# a.groupby(['hospital_expire_flag']).count()['row_id'].plot(kind='barh')

patients = pd.read_csv("/Desktop/data/uplift_related/mimic-iii-clinical-database-1.4/PATIENTS.csv")
patients.columns = map(str.lower, patients.columns)
ap = pd.merge(a, patients, on='subject_id', how='inner')
# ap.groupby(['religion','gender']).size().unstack().plot(kind="barh", stacked=True)

icustays = pd.read_csv("/Desktop/data/uplift_related/mimic-iii-clinical-database-1.4/ICUSTAYS.csv")
icustays.columns = map(str.lower, icustays.columns)
api = pd.merge(ap, icustays, on='hadm_id', how='inner')

services = pd.read_csv("/Desktop/data/uplift_related/mimic-iii-clinical-database-1.4/SERVICES.csv")
services.columns = map(str.lower, services.columns)
apis = pd.merge(api, services, on='hadm_id', how='inner', suffixes=('', '_new'))

print(apis)
print(apis.columns)

apis.to_csv('~/Downloads/mimic_iii_uplift_data.csv')

# drgcodes = pd.read_csv("~/Downloads/mimic-iii-clinical-database-demo-1.4/DRGCODES.csv")
# drgcodes.columns = map(str.lower, drgcodes.columns)
# appd = pd.merge(app, drgcodes, on='subject_id', how='inner', suffixes=('', '_new'))
#
# print(appd.columns)
# # appd.groupby(['drg_type']).count()['row_id'].plot(kind='barh')
# print(appd)

# c = pd.read_csv("~/Downloads/mimic-iii-clinical-database-demo-1.4/CPTEVENTS.csv")
# c.columns = map(str.lower, c.columns)
# ac = pd.merge(a, c, on = 'hadm_id' , how = 'inner')
# ac.groupby(['discharge_location','sectionheader']).size().unstack().plot(kind="barh", stacked=True)
#
# print(ac)

plt.show()
