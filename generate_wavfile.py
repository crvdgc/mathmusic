from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd


# we want to generate a wave

def genwave(duration=1.0, start=0.0, end=1.0, freq=440.00, sample_rate=44100):
    # generate a segmented wave, like this  -----------/\/\/\/\/\/\----------
    #                                       |<---------duration------------>|
    #                                       |          ↑start     ↑end      |
    #
    sample_points = np.arange(0, end - start, 1/sample_rate)
    wave = np.sin(freq * sample_points * 2 * np.pi) * 32767
    samples = np.zeros(int(duration * sample_rate))
    samples[int(start * sample_rate):int(end * sample_rate)] = wave
    return samples


# samples = genwave()
# sample_rate= 44100
# # save the wave file
# wavfile.write('wavfiles/test.wav', sample_rate, samples)


tonesscale = [0,2,4,5,7,9,11,12]

from temperament import twelve_tone_equal
pitches = twelve_tone_equal()

sample_rate = 44100

samples = np.zeros(int(len(tonesscale) * sample_rate / 2))
for i in range(len(tonesscale)):
    note = tonesscale[i]
    new_wave = genwave(duration=len(tonesscale)/2, start=i /
                       2, end=(i+1)/2, freq=pitches[note+60])
    samples = samples + new_wave

# smooth the song to avoid noise between two note
samples = pd.DataFrame(samples)
samples = samples.rolling(2).mean()
samples = np.array(samples)[2:]

wavfile.write('wavfiles/scale.wav', sample_rate, samples)
