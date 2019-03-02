import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

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

# Building ANN
ann = Sequential()
ann.add(Dense(input_dim=X_train.shape[1], output_dim=6, init='uniform', activation='relu'))
ann.add(Dense(output_dim=6, init='uniform', activation='relu'))
ann.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting to training data
ann.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting test data
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)
# Evaluating confusion matrix
metrics = confusion_matrix(y_test, y_pred)
print(metrics)