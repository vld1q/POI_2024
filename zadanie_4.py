import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

if __name__ == "__main__":
    data_frame = pd.read_csv('feature_vectors.csv')
    v_feature = data_frame.iloc[:, 2:].values
    v_content = data_frame.iloc[:, 0].values
    y_int = LabelEncoder().fit_transform(v_content)
    y_int = y_int.reshape(len(y_int), 1)
    y_onehot = OneHotEncoder(sparse_output=False).fit_transform(y_int)

    # Podział zbioru na część treningową i testową
    X_train, X_test, y_train, y_test = train_test_split(v_feature, y_onehot, test_size=0.3, random_state=42)

    # Tworzenie modelu sieci neuronowej
    model = Sequential()
    # design
    model.add(Dense(units=16, activation='sigmoid', input_dim=X_train.shape[1]))
    model.add(Dense(units=y_onehot.shape[1], activation='softmax'))
    # train
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=128, batch_size=16, shuffle=True)

    # calc confusion matrix
    y_expected = model.predict(X_test)
    y_expected_int = np.argmax(y_expected, axis=1)
    y_test_int = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test_int, y_expected_int)

    print("confusion matrix")
    for i, row in enumerate(cm):
        row = ", ".join([f"{n}" for n in row])
        print(f"{i+1}:\t\t{row}")