# import tensorflow
# import theano
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

file_path = r'.\Churn_Modelling.csv'
df = pd.read_csv(file_path)

# Split dependent/independent variables
X = df.iloc[:, 3:13].values
y = df.iloc[:, 13].values

# Encode categorical variables
encoder_geography = LabelEncoder()
encoder_gender = LabelEncoder()
X[:, 1] = encoder_geography.fit_transform(X[:, 1])
X[:, 2] = encoder_gender.fit_transform(X[:, 2])

one_hot_geography = OneHotEncoder(categorical_features=[1])
X = one_hot_geography.fit_transform(X).toarray()
X = X[:, 1:]  # drop first dummy variable

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pass