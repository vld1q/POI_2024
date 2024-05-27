import os.path
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Wczytanie danych z pliku CSV
cwd = os.getcwd()
feature_vectors_file = os.path.sep.join([f"{cwd}", "data", "out", "feature_vectors.csv"])
data_frame = pd.read_csv(feature_vectors_file)

converter_lambda = lambda x: float(x.strip('[]').split()[0])
for label in "Dissimilarity Correlation Contrast Energy Homogeneity ASM".split():
    data_frame[label] = data_frame[label].apply(converter_lambda)

X = data_frame.drop(['Category', 'File'], axis=1)
Y = data_frame['Category']

class_labels = Y.unique()
clf = SVC(gamma='auto')  # SVC identifier

# training and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, stratify=Y)
clf.fit(x_train, y_train)
y_expected = clf.predict(x_test)
acc = accuracy_score(y_test, y_expected)
print("Accuracy:", acc)

# gen confusion matrix
cm = confusion_matrix(y_test, y_expected, normalize='true')
print("Confusion Matrix:")
for i, row in enumerate(cm):
    val: str = ", ".join([f"{n}" for n in row])
    print(f"{i}:\t\t{val}")

display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
display.plot(cmap=plt.cm.Blues)
plt.show()
