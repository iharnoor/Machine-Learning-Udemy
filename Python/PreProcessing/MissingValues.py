# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

dataset = pd.read_csv('Data.csv')
# print(dataset)
X = dataset.iloc[:, :-1].values  # everything except the last column
# print(X)
y = dataset.iloc[:, 3].values  # just the last column
# Fixing the Missing Data
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)  # object created
imputer = imputer.fit(X[:, 1:3])  # fitting columns 1 and 2 containing missing values
# matrix is added to the imputer object
X[:, 1:3] = imputer.transform(X[:, 1:3])  # replacing missing by the mean
print(X)
