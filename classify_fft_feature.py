from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from scipy.stats.stats import pearsonr
import os
import itertools
from functools import reduce
from sklearn.neighbors import KNeighborsClassifier

from feature_extraction import features_octave_merge
import temperament


def freq_to_note(frequencies):
    return np.log2(frequencies/440) * 12 + 69


def distance(vec_a, vec_b):
    vec_a = np.abs(freq_to_note(vec_a) - np.rint(freq_to_note(vec_a)))
    vec_b = np.abs(freq_to_note(vec_b) - np.rint(freq_to_note(vec_b)))
    return np.log(np.dot(vec_a, vec_b))


def calc_features():
    # calc features for all wavefiles in training set
    temps = ['Just_Intonation', 'Pythagorean', 'Twelve_Tone_Equal']
    features = dict()
    for temp in temps:
        # features[temp] = dict()
        features[temp] = list()
        for wavfile_name in os.listdir('wavfiles/training_set/{temp}/'.format(temp=temp)):
            wavfile_path = 'wavfiles/training_set/{temp}/{wavfile_name}'.format(
                temp=temp, wavfile_name=wavfile_name)
            print('extracting features for ' + wavfile_path)
            sample_rate, samples = wavfile.read(wavfile_path)
            # features[temp][wavfile_name] = features_octave_merge(
            # samples, sample_rate, first_n=8, nperseg=4096, nfft=32768)
            features[temp].append(features_octave_merge(
                samples, sample_rate, first_n=8, nperseg=4096, nfft=32768))
    return features


def calc_corrs(features):
    # calculate table between two temps and draw it
    dataset_size = len(features['Just_Intonation'])

    corrs = np.zeros((3*(dataset_size),
                      3*(dataset_size)))

    all_features = features['Just_Intonation'] + \
        features['Pythagorean'] + features['Twelve_Tone_Equal']

    for i, vec_a in enumerate(all_features):
        for j, vec_b in enumerate(all_features):
            print('calculating distance for', i, j)
            corrs[i, j] = distance(vec_a, vec_b)

    return corrs


features = calc_features()
corrs = calc_corrs(features)
all_features = features['Just_Intonation'] + \
    features['Pythagorean'] + features['Twelve_Tone_Equal']

# corrs = np.log(corrs)

# plt.subplots(figsize=(10, 8))
# plt.pcolormesh(corrs, cmap='gist_ncar')
# plt.colorbar()
# plt.show()


def calc_dists(samples, sample_rate):
    features = features_octave_merge(
        samples, sample_rate, first_n=8, nperseg=4096, nfft=32768)

    # calculate the distance form all training data
    dists_sample = np.zeros(len(corrs))
    for i, features_training in enumerate(all_features):
        dists_sample[i] = distance(features_training, features)

    return dists_sample


def classifier_simple_inner(samples, sample_rate, bins=3):
    dists_sample = calc_dists(samples, sample_rate)
    # second layer: calculate correlation of the characteristic dists and all corrs we get before
    # to determine probability
    corr_sample = np.zeros(len(corrs))
    for i in range(len(corrs)):
        corr_dist = np.abs(corrs[i, :] - dists_sample)
        corr_sample[i] = np.inner(corr_dist, corr_dist)

    # return corr_sample

    # integrate over traning set and normalize to get total probability
    probs = np.zeros(bins)
    for i in range(bins):
        probs[i] = sum(corr_sample[(len(corr_sample) // bins)
                                   * i: (len(corr_sample) // bins) * (i+1)])

    return 1 / probs


def classifier_knn_init(bins, n_neighbors):
    global knn, knn_init
    print('initializing knn model')
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # prepare vectors and labels
    label = np.concatenate(
        [np.ones(corrs.shape[0]//bins, dtype=int) * i for i in range(bins)])

    # Train the model using the training sets
    knn.fit(corrs, label)
    knn_init = True
    print('knn model: ', knn)


knn_init = False


def classifier_knn(samples, sample_rate, bins=3, n_neighbors=50):
    global knn_init
    if knn_init:
        pass
    else:
        classifier_knn_init(bins, n_neighbors)
    # Predict Output
    # wavfile_name = 'wavfiles/test_set/Just_Intonation/mz_330_1.wav.30.wav'
    # sample_rate, samples = wavfile.read(wavfile_name)
    dists_sample = calc_dists(samples, sample_rate)
    dists_sample = dists_sample.reshape(1, -1)
    predicted = knn.predict(dists_sample)  # 0:Overcast, 2:Mild
    ans = np.zeros(bins)
    ans[predicted[0]] = 1.0
    return ans


def validate(clf, **kwargs):
    # calc features for all wavefiles in validation set and classify them to see
    # is our method working
    temps = ['Just_Intonation', 'Pythagorean', 'Twelve_Tone_Equal']
    answers = dict()
    for temp in temps:
        answers[temp] = list()
        for wavfile_name in os.listdir('wavfiles/validation_set/{temp}/'.format(temp=temp)):
            wavfile_path = 'wavfiles/validation_set/{temp}/{wavfile_name}'.format(
                temp=temp, wavfile_name=wavfile_name)
            print('classifying ' + wavfile_path + ', result:', end='')
            sample_rate, samples = wavfile.read(wavfile_path)
            ans = clf(samples, sample_rate, **kwargs)
            print(ans, temps[ans.argmax()])
            answers[temp].append(ans)
    return answers

# validate(clf=classifier_simple_inner)


def test_for_correct_rate(clf, **kwargs):
    # this is basically the same with validate, except for some directory differences
    # calc features for all wavefiles in validation set and classify them to see
    # is our method working
    temps = ['Just_Intonation', 'Pythagorean', 'Twelve_Tone_Equal']
    answers = dict()
    correct = dict()
    wrong = dict()
    for temp in temps:
        answers[temp] = list()
        correct[temp] = 0
        wrong[temp] = 0
        for wavfile_name in os.listdir('wavfiles/test_set/{temp}/'.format(temp=temp)):
            wavfile_path = 'wavfiles/test_set/{temp}/{wavfile_name}'.format(
                temp=temp, wavfile_name=wavfile_name)
            print('classifying ' + wavfile_path + ', result:', end='')
            sample_rate, samples = wavfile.read(wavfile_path)
            ans = clf(samples, sample_rate, **kwargs)
            print(ans, temps[ans.argmax()], end='')
            if temps[ans.argmax()] == temp:
                correct[temp] += 1
                print(', correct')
            else:
                wrong[temp] += 1
                print(', wrong')
            answers[temp].append(ans)
    return answers, correct, wrong

knn_param_test = dict()
for i in range(1, 50):
    knn_init = False
    answers_knn, correct_knn, wrong_knn = test_for_correct_rate(clf=classifier_knn, n_neighbors=i)
    print(correct_knn, wrong_knn)
    knn_param_test[i] = (correct_knn, wrong_knn)

print(knn_param_test)

# n_neighbors = 10:
# {'Just_Intonation': 44, 'Pythagorean': 43, 'Twelve_Tone_Equal': 42} 
# {'Just_Intonation': 6, 'Pythagorean': 7, 'Twelve_Tone_Equal': 8}

# n_neighbors = 20:
#{'Just_Intonation': 43, 'Pythagorean': 43, 'Twelve_Tone_Equal': 43} 
# {'Just_Intonation': 7, 'Pythagorean': 7, 'Twelve_Tone_Equal': 7}

# n_neighbors = 30:
# {'Just_Intonation': 42, 'Pythagorean': 41, 'Twelve_Tone_Equal': 44} 
# {'Just_Intonation': 8, 'Pythagorean': 9, 'Twelve_Tone_Equal': 6}

# n_neighbors = 50:
# {'Just_Intonation': 42, 'Pythagorean': 42, 'Twelve_Tone_Equal': 43} 
# {'Just_Intonation': 8, 'Pythagorean': 8, 'Twelve_Tone_Equal': 7}


# answers_simple_inner, correct_simple_inner, wrong_simple_inner = test_for_correct_rate(
#     clf=classifier_simple_inner)
# print(correct_simple_inner, wrong_simple_inner)
# {'Just_Intonation': 45, 'Pythagorean': 42, 'Twelve_Tone_Equal': 42} 
# {'Just_Intonation': 5, 'Pythagorean': 8, 'Twelve_Tone_Equal': 8}
