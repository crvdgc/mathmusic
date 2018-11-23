from sklearn import svm
import numpy as np

def predict_svm(dataset):
    train = np.concatenate((dataset.train, dataset.valid), axis=1)
    X =

