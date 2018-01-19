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

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_X = LabelEncoder()  # object created
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])  # transformed 1st column saved in 1st column
# Transforming Country column: Splitting in 3 columns
one_hot_encoder = OneHotEncoder(categorical_features=[0])  # transforming the first column
X = one_hot_encoder.fit_transform(X).toarray()
print(X)
# Transforming y
label_encoder_y = LabelEncoder()
y= label_encoder_y.fit_transform(y)
print(y)