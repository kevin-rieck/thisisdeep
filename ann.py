import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


def load_data(file_path):
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
    return X_train, X_test, y_train, y_test


def build_classifier(X_train):
    # Building ANN
    ann = Sequential()
    ann.add(Dense(input_dim=X_train.shape[1], output_dim=6, init='uniform', activation='relu'))
    ann.add(Dense(output_dim=6, init='uniform', activation='relu'))
    ann.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return ann


def predict_without_kfold(X_train, X_test, y_train, y_test):
    classifier = build_classifier(X_train)
    # Fitting to training data
    classifier.fit(X_train, y_train, batch_size=10, epochs=100)

    # Predicting test data
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    # Evaluating confusion matrix
    metrics = confusion_matrix(y_test, y_pred)
    print(metrics)


def run_kfold(X_train, y_train):
    classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100, **{'X_train': X_train})
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
    return np.mean(accuracies), np.std(accuracies)


if __name__ == '__main__':
    file_path = r'.\Churn_Modelling.csv'
    X_train, X_test, y_train, y_test = load_data(file_path)
    #predict_without_kfold(X_train, X_test, y_train, y_test)
    mean_acc, std_acc = run_kfold(X_train, y_train)
    print('Mean: {:.3f}, StDev: {:.3f}'.format(mean_acc, std_acc))