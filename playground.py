from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd


from spectrogram import draw_log_scale_spectrogram_12tone


def get_fft(wavfile_name):

    sample_rate, samples = wavfile.read(wavfile_name)

    sp = np.fft.rfft(samples)
    freq = np.fft.rfftfreq(samples.shape[-1], 1/sample_rate)

    spi = sp.imag

    spi2 = np.power(spi, 2)

    spi2 = spi2 / spi2.max()

    return freq, spi2

a = get_fft('wavfiles/scale_Pythagorean.wav')
b = get_fft('wavfiles/scale_Just_intonation.wav')
c = get_fft('wavfiles/scale_twelve_tone_equal.wav')
d = get_fft('wavfiles/sample_10s.wav')

# plt.plot(freq, sp.real, freq, sp.imag)
# plt.plot(freq, sp.real)

plt.plot(*d)

plt.show()