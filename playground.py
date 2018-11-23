from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd


from spectrogram import draw_log_scale_spectrogram_12tone

wavfile_name = 'wavfiles/scale.wav'

sample_rate, samples = wavfile.read(wavfile_name)

sp = np.fft.rfft(samples)
freq = np.fft.rfftfreq(samples.shape[-1], 1/sample_rate)

# plt.plot(freq, sp.real, freq, sp.imag)
# plt.plot(freq, sp.real)
plt.plot(freq, sp.imag)

plt.show()