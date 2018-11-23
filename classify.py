from mfcc import mfcc_features
import numpy as np


class Dataset:
    def __init__(self, data):
        n = data.shape[0]
        trainI = int(n * 0.8)
        validI = trainI + int(n * 0.1)
        (self.train, self.valid, self.test) = np.split(data, [trainI, validI])

    def correct_rate(self, predict):
        testY = [t[1] for t in self.test]
        count = sum([1 if prey == y else 0 for (prey, y) in zip(predict, testY)])
        return count / len(testY)

def svm_predict(dataset):
    