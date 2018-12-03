from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from scipy.stats.stats import pearsonr

from feature_extraction import features_octave_merge
import temperament


def freq_to_note(frequencies):
    return np.log2(frequencies/440) * 12 + 69


def distance(vec_a, vec_b):
    # return np.correlate(vec_a, vec_b)

    # return pearsonr(vec_a, vec_b)[0]

    # return np.dot(vec_a, vec_b)

    # vec_a = freq_to_note(vec_a) - np.rint(freq_to_note(vec_a))
    # vec_b = freq_to_note(vec_b) - np.rint(freq_to_note(vec_b))
    # return np.dot(vec_a, vec_b)

    vec_a = np.abs(freq_to_note(vec_a) - np.rint(freq_to_note(vec_a)))
    vec_b = np.abs(freq_to_note(vec_b) - np.rint(freq_to_note(vec_b)))
    return np.dot(vec_a, vec_b)

    # vec_a = np.power(freq_to_note(vec_a) - np.rint(freq_to_note(vec_a)), 2)
    # vec_b = np.power(freq_to_note(vec_b) - np.rint(freq_to_note(vec_b)), 2)
    # return np.dot(vec_a, vec_b)

    # vec_a = np.abs(freq_to_note(vec_a) - np.rint(freq_to_note(vec_a)))
    # vec_b = np.abs(freq_to_note(vec_b) - np.rint(freq_to_note(vec_b)))
    # return pearsonr(vec_a, vec_b)[0]


# get our features for 3 temps
temps = ['Just_Intonation', 'Pythagorean', 'Twelve_Tone_Equal']
features = dict()
for temp in temps:
    features[temp] = list()
    for i in range(0, 840, 30):
        wavfile_name = 'wavfiles/'+temp+'/mzt_331_1.wav.{i}.wav'.format(i=i)
        print('extracting features for ' + wavfile_name)
        sample_rate, samples = wavfile.read(wavfile_name)
        a = features_octave_merge(
            samples, sample_rate, first_n=8, nperseg=4096, nfft=32768)
        features[temp].append(a)

# get our features for 3 temps
temps = ['Just_Intonation', 'Pythagorean', 'Twelve_Tone_Equal']
features3 = dict()
for temp in temps:
    features3[temp] = list()
    for i in range(0, 210, 30):
        wavfile_name = 'wavfiles/'+temp+'/mzt_331_3.wav.{i}.wav'.format(i=i)
        print('extracting features for ' + wavfile_name)
        sample_rate, samples = wavfile.read(wavfile_name)
        a = features_octave_merge(
            samples, sample_rate, first_n=8, nperseg=4096, nfft=32768)
        features3[temp].append(a)


# calculate table between two temps and draw it
dataset_size = len(features['Just_Intonation'])
dataset3_size = len(features3['Just_Intonation'])

corrs = np.zeros((3*(dataset_size + dataset3_size),
                  3*(dataset_size + dataset3_size)))

all_features = features['Just_Intonation'] + \
    features['Twelve_Tone_Equal'] + features['Pythagorean'] + features3['Just_Intonation'] + \
    features3['Twelve_Tone_Equal'] + features3['Pythagorean']

for i, vec_a in enumerate(all_features):
    for j, vec_b in enumerate(all_features):
        print('calculating distance for', i, j)
        corrs[i, j] = distance(vec_a, vec_b)

corrs = np.log(corrs)
plt.subplots(figsize=(10, 8))
plt.pcolormesh(corrs, cmap='nipy_spectral')
plt.colorbar()
plt.show()
