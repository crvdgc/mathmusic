import random
import numpy as np
import os
import mfcc
from sklearn import svm
import matplotlib.pyplot as plt

top_dir = r'./wavfiles/'
dir_names = [r'Just_Intonation', r'Pythagorean', r'Twelve_Tone_Equal']
label_num = len(dir_names)


class Dataset:
    def __init__(self, data):
        n = data.shape[0]
        trainIndex = int(n * 0.8)
        validIndex = trainIndex + int(n * 0.1)
        while not all([any([y == l for (x, y) in data[:trainIndex]]) for l in range(label_num)]):
            random.shuffle(data)
        (self.train, self.valid, self.test) = np.split(data, [trainIndex, validIndex])

    def train_and_valid(self):
        return np.concatenate((self.train, self.valid), axis=0)


def sieve_label(xys, l):
    return np.array([x for (x, y) in xys if y == l])


def accuracy(predict, truth):
    right = sum([1 if prey == y else 0 for (prey, y) in zip(predict, truth)])
    return right / len(truth)


def split(xys):
    return np.array([x for (x, y) in xys]), np.array([y for (x, y) in xys])


# Parzen window method
def pn_generator(hn, kernel, x_proto):
    return lambda x: sum([kernel((x - xi) / hn) for xi in x_proto]) / (x_proto.shape[0] * hn)


# window kernel function
def phi(x):
    # print(x.shape)
    if np.sqrt(np.inner(x, x)) <= 1 / 2:
        return 1
    else:
        return 0


# gaussian kernel
def gauss(x):
    return np.exp(-np.inner(x, x) / 2) / np.sqrt(2 * np.pi)


# k-nearest neighbour
def pkn_generator(kn, x_proto):
    def predicator(x):
        distances = [np.sqrt(np.inner(x - xi, x - xi)) for xi in x_proto]
        np.sort(distances)
        return kn / (distances[kn-1] ** x_proto.shape[1] * x_proto.shape[0])
    return predicator


# sample_points = [10 * 2**(-i/12) for i in range(0, 11 * 12)]


# def kl_divergence(p, q):
#     return sum([p(x) * np.log(p(x)/q(x)) if q(x) > 0 else 0 for x in sample_points])


def do_test(model, test):
    txs, tys = split(test)
    return accuracy([model(tx) for tx in txs], tys)


def select_param(ps, f, train, valid):
    acc = [do_test(f(p, train), valid) for p in ps]
    return ps[acc.index(max(acc))]


if __name__ == '__main__':

    raw_data = []
    for l in range(label_num):
        cur_dir = top_dir + dir_names[l]
        raw_data += [[mfcc.mfcc_features(cur_dir+r'/'+f), l] for f in os.listdir(cur_dir)[:15]]
    dataset = Dataset(np.array(raw_data))

    # parzen
    def partial_parzen_window_gen(kernel):
        def partial_parzen_window(hn, train):
            models = [pn_generator(hn, kernel, sieve_label(train, l)) for l in range(label_num)]

            def predictor(x):
                probs = [m(x) for m in models]
                return probs.index(max(probs))
            return predictor
        return partial_parzen_window

    parzen_hns = range(10, 110, 10)
    # parzen, window kernel
    partial_phi = partial_parzen_window_gen(phi)
    hn = select_param(parzen_hns, partial_phi, dataset.train, dataset.valid)
    model_phi = partial_phi(hn, dataset.train_and_valid())
    acc_phi = do_test(model_phi, dataset.test)
    print('Parzen, phi\naccuracy: %s\n\n' % acc_phi)

    # parzen, gauss kernel
    partial_gauss = partial_parzen_window_gen(gauss)
    hn = select_param(parzen_hns, partial_gauss, dataset.train, dataset.valid)
    model_gauss = partial_gauss(hn, dataset.train_and_valid())
    acc_gauss = do_test(model_gauss, dataset.test)
    print('Parzen, gauss\naccuracy: %s\n\n' % acc_gauss)


    # knn
    def partial_knn(kn, train):
        models = [pkn_generator(kn, sieve_label(train, l)) for l in range(label_num)]

        def predicator(x):
            probs = [m(x) for m in models]
            return probs.index(max(probs))
        return predicator

    kn = select_param(range(2, 5), partial_knn, dataset.train, dataset.valid)
    model_knn = partial_knn(kn, dataset.train_and_valid())
    acc_knn = do_test(model_knn, dataset.test)
    print('KNN\naccuracy: %s\n\n' % acc_knn)

    # svm
    dx, dy = split(dataset.train_and_valid())
    clf = svm.SVC(gamma='auto')
    clf.fit(dx, dy)
    tx, ty = split(dataset.test)
    predict_svm = clf.predict(tx)
    print(predict_svm)
    print(ty)
    acc_svm = accuracy(predict_svm, ty)
    print('SVM\naccuracy: %s' % acc_svm)
