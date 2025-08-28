#1- NAME: Bodhu Shravya UID: U01122915 EMAIL:bodhu.2@wright.edu 
#2- NAME: Lohitha Donuri UID: U01125638 EMAIL: donuri.3@wright.edu
#3- NAME: Niharika Kanugovi UID: U01108474 EMAIL: Kanugovi.2@wright.edu
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score
CONFIG = {
    "knn_k": 1,
    "svm_random_state": 42,
    "input_files": {
        "train_data": "trainMatrixModified.txt",
        "train_labels": "trainClasses.txt",
        "test_data": "testMatrixModified.txt",
        "test_labels": "testClasses.txt"
    }
}
def load_data():
    X_train = np.loadtxt(CONFIG["input_files"]["train_data"]).T
    X_test = np.loadtxt(CONFIG["input_files"]["test_data"]).T
    y_train = np.loadtxt(CONFIG["input_files"]["train_labels"])[:, 1]
    y_test = np.loadtxt(CONFIG["input_files"]["test_labels"])[:, 1]    
    return X_train, y_train, X_test, y_test
def main():
    X_train, y_train, X_test, y_test = load_data()
    # kNN
    knn = KNeighborsClassifier(n_neighbors=CONFIG["knn_k"], metric='cosine')
    knn.fit(X_train, y_train)
    np.savetxt('kNN-CF.txt', confusion_matrix(y_test, knn.predict(X_test)), fmt='%d')
    with open('kNN-Metric.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy_score(y_test, knn.predict(X_test)):.3f}')
    # SVM
    svm = LinearSVC(random_state=CONFIG["svm_random_state"])
    svm.fit(X_train, y_train)
    np.savetxt('SVM-CF.txt', confusion_matrix(y_test, svm.predict(X_test)), fmt='%d')
    with open('SVM-Metric.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy_score(y_test, svm.predict(X_test)):.3f}')
if __name__ == "__main__":
    main()
    print("Output files generated and saved in the same directory.")