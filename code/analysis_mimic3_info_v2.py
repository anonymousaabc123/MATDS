import pandas as pd
import pandas.io.sql as psql
# plot a figure directly on Notebook
import matplotlib.pyplot as plt

admission_type_dict = {'ELECTIVE': 1, 'URGENT': 2, 'EMERGENCY': 2, 'NEWBORN': 3}
admission_location_dict = {'TRSF WITHIN THIS FACILITY': 1, 'TRANSFER FROM SKILLED NUR': 2, 'TRANSFER FROM OTHER HEALT': 3,
                           'TRANSFER FROM HOSP/EXTRAM': 4, 'PHYS REFERRAL/NORMAL DELI': 5, 'HMO REFERRAL/SICK': 6,
                           'EMERGENCY ROOM ADMIT': 7, 'CLINIC REFERRAL/PREMATURE': 8, '** INFO NOT AVAILABLE **': 9}
insurance_dict = {'Medicare': 1, 'Private': 2, 'Medicaid': 3, 'Government': 4, 'Self Pay': 5}
religion_dict = {'': 0, 'BUDDHIST': 1, 'CATHOLIC': 2, 'JEWISH': 3, 'MUSLIM': 4, 'HINDU': 5, 'OTHER': 6, 'METHODIST': 7,
                 'EPISCOPALIAN': 8, 'UNOBTAINABLE': 9, 'PROTESTANT QUAKER': 10, 'CHRISTIAN SCIENTIST': 11, 'BAPTIST': 12,
                 'NOT SPECIFIED': 13, '7TH DAY ADVENTIST': 14, 'ROMANIAN EAST. ORTH': 15, 'HEBREW': 16, "JEHOVAH'S WITNESS": 17,
                 'GREEK ORTHODOX': 18, 'UNITARIAN-UNIVERSALIST': 19, 'LUTHERAN': 20}
marital_status_dict = {'': 0, 'MARRIED': 1, 'DIVORCED': 2, 'SINGLE': 3, 'SEPARATED': 4, 'LIFE PARTNER': 5, 'UNKNOWN (DEFAULT)': 6, 'WIDOWED': 7}
ethnicity_dict = {'AMERICAN INDIAN/ALASKA NATIVE': 0, 'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE': 0, 'ASIAN': 1,
                  'ASIAN - ASIAN INDIAN': 1, 'ASIAN - CAMBODIAN': 1, 'ASIAN - CHINESE': 1, 'ASIAN - FILIPINO': 1, 'ASIAN - JAPANESE': 1,
                  'ASIAN - KOREAN': 1, 'ASIAN - OTHER': 1, 'ASIAN - THAI': 1, 'ASIAN - VIETNAMESE': 1, 'BLACK/AFRICAN': 2,
                  'BLACK/AFRICAN AMERICAN': 2, 'BLACK/CAPE VERDEAN': 2, 'BLACK/HAITIAN': 2, 'HISPANIC OR LATINO': 3,
                  'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)': 3, 'HISPANIC/LATINO - COLOMBIAN': 3, 'HISPANIC/LATINO - CUBAN': 3,
                  'HISPANIC/LATINO - DOMINICAN': 3, 'HISPANIC/LATINO - GUATEMALAN': 3, 'HISPANIC/LATINO - HONDURAN': 3, 'HISPANIC/LATINO - MEXICAN': 3,
                  'HISPANIC/LATINO - PUERTO RICAN': 3, 'HISPANIC/LATINO - SALVADORAN': 3, 'CARIBBEAN ISLAND': 4, 'MIDDLE EASTERN': 4,
                  'MULTI RACE ETHNICITY': 4, 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 4, 'PATIENT DECLINED TO ANSWER': 4, 'PORTUGUESE': 4,
                  'SOUTH AMERICAN': 4, 'OTHER': 5, 'UNABLE TO OBTAIN': 5, 'UNKNOWN/NOT SPECIFIED': 5, 'WHITE': 6, 'WHITE - BRAZILIAN': 6,
                  'WHITE - EASTERN EUROPEAN': 6, 'WHITE - OTHER EUROPEAN': 6, 'WHITE - RUSSIAN': 6}
gender_dict = {'M': 1, 'F': 2}

a = pd.read_csv("/Downloads/mimic_iii_uplift_data_v2.csv")
a.columns = map(str.lower, a.columns)
a['admission_type'] = a['admission_type'].map(admission_type_dict)
a['admission_location'] = a['admission_location'].map(admission_location_dict)
a['insurance'] = a['insurance'].map(insurance_dict)
a['religion'] = a['religion'].map(religion_dict)
a['marital_status'] = a['marital_status'].map(marital_status_dict)
a['ethnicity'] = a['ethnicity'].map(ethnicity_dict)
a['gender'] = a['gender'].map(gender_dict)
a = a.fillna(value=0)
#
# a.to_csv("/Downloads/mimic_iii_uplift_data_v3.csv")

a = pd.read_csv("/Downloads/mimic_iii_uplift_data_v3.csv")
a.groupby(['curr_service']).count()['row_id'].plot(kind='barh')
curr_service_dict = {'CMED': 1, 'CSURG': 2, 'DENT': 3, 'ENT': 4, 'GU': 5, 'GYN': 6, 'MED': 7, 'NB': 8, 'NBB': 9, 'NMED': 10,
                     'NSURG': 11, 'OBS': 12, 'OMED': 13, 'ORTHO': 14, 'PSURG': 15, 'PSYCH': 16, 'SURG': 17, 'TRAUM': 18, 'TSURG': 19, 'VSURG': 20}
plt.show()
a['curr_service'] = a['curr_service'].map(curr_service_dict)
a.to_csv("/Downloads/Mimic_iii_uplift_data_ver2.csv")

# print(sorted(set(a['gender'])))
# a.groupby(['gender']).count()['row_id'].plot(kind='barh')
# plt.show()

# a = pd.read_csv("/Downloads/mimic_iii_uplift_data_v3.csv")
# curr_service_dict = {'CSURG': 0, 'CMED': 1}
# a['curr_service'] = a['curr_service'].map(curr_service_dict)
# a_part = a.loc[a['curr_service'].isin([0, 1])]
# # a_part.groupby(['curr_service']).count()['row_id'].plot(kind='barh')
# # a_part.groupby(['curr_service', 'hospital_expire_flag']).size().unstack().plot(kind="barh", stacked=True)
# # plt.show()
# a_part.to_csv("/Downloads/Mimic_iii_uplift_data_ver1.csv")

# a = pd.read_csv("/Downloads/Mimic_iii_uplift_data_ver1.csv")
# a.groupby(['curr_service', 'hospital_expire_flag']).size().unstack().plot(kind="barh", stacked=True)
# plt.show()


