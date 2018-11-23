from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

from temperament import twelve_tone_equal

# we want to generate a wave 

def genwave(duration=1.0, start=0.0, end=1.0, freq=440.00, sample_rate=44100):
    # generate a segmented wave, like this  -----------/\/\/\/\/\/\----------
    #                                       |<---------duration------------>|
    #                                       |          ↑start     ↑end      |
    #
    sample_points = np.arange(0, end - start, 1/sample_rate)
    wave = np.sin(freq * sample_points * 2 * np.pi)
    samples = np.zeros(int(duration * sample_rate))
    samples[int(start * sample_rate):int(end * sample_rate)] = wave
    return samples

samples = genwave()
sample_rate= 44100
# save the wave file
wavfile.write('wavfiles/test.wav', sample_rate, samples)