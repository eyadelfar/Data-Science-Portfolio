##
# import all useful libraries:
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

##
# Load the data from the text file

HPC = pd.read_csv('02 Household Power Consumption.txt', sep=';')

HPC.head()  # checking first 5 columns
HPC.shape  # checking the shape (185711, 9)

# Cleaning data:


df_obj = HPC.select_dtypes(['object'])  # Making object that selects only strings

HPC[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())  # lambda to strip strings

HPC = HPC.replace(dict.fromkeys(['', '?'], np.nan))  # replacing missing data with nans

# Making sure that python knows the exact datatypes to be able to impute clean
for i in HPC.columns[2:]:  # Excluding date and time
    HPC.loc[:, i].astype(float)  # Defining the format

# Now we can check number of nans and remove them
print(HPC.isnull().sum().sum())
HPC = HPC.drop(['Date', 'Time'], axis=1)  # we wont use time series
imputer = SimpleImputer(strategy='mean')
HPC = pd.DataFrame(imputer.fit_transform(HPC), columns=HPC.columns)
print(HPC.isnull().sum().sum())

# Feature Select , extract

# (global_active_power*1000/60 - sub_metering_1 - sub_metering_2 - sub_metering_3) = active energy per minute
HPC['active_energy_per_minute'] = ((HPC['Global_active_power'] * 1000 / 60) -
                                   HPC['Sub_metering_1'] -
                                   HPC['Sub_metering_2'] -
                                   HPC['Sub_metering_3'])
HPC = HPC.drop(['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], axis=1)

# Time to scale !

Scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
Scaler = Scaler.fit_transform(HPC)

HPC_Scaled = pd.DataFrame(Scaler, columns=HPC.columns)
pd.set_option('display.max_columns', None)
print(HPC_Scaled.head(5))

# No time series clustering:

X_train, X_test = train_test_split(HPC, test_size=0.25, random_state=44, shuffle=True)
Model = KMeans(n_clusters=9)
Model.fit(X_train)

print('KMeansModel Train Score is : ', Model.score(X_train))
print('KMeansModel Test Score is : ', Model.score(X_test))

print('KMeansModel Train Score is : ', round(silhouette_score(X_train, Model.labels_, metric='euclidean'), 3))
