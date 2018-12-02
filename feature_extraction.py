# we can use multiple methods to extract features,
# since our main goal is to analysis the temperament system,
# using simple max[ fft [ f ] ] should be fine

from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd


def normed_fft2(samples, sample_rate, **kwargs):
    # we take the real FFT of samples, so we only need the imag part.
    # since there will be negative values which might cause the peak
    # detection to fail at rapid oscillating areas, we can take the
    # square of the spectrum. Finally, since samples read from file
    # can have vary different absolute values, we also should normalize
    # the spectrum by scaling max to 1.

    # takes     samples: np_array,  sample_rate: int
    # returns   freq: np_array,     spi2: np_array
    sp = np.fft.rfft(samples, **kwargs)
    freq = np.fft.rfftfreq(samples.shape[-1], 1/sample_rate)

    spi = sp.imag

    spi2 = np.power(spi, 2)

    spi2 = spi2 / spi2.max()

    return freq, spi2


def normed_abs_fft(samples, sample_rate, **kwargs):
    # we take the real FFT of samples, so we only need the imag part.
    # since there will be negative values which might cause the peak
    # detection to fail at rapid oscillating areas, we can take the
    # square of the spectrum. Finally, since samples read from file
    # can have vary different absolute values, we also should normalize
    # the spectrum by scaling max to 1.

    # takes     samples: np_array,  sample_rate: int
    # returns   freq: np_array,     spi2: np_array
    sp = np.fft.rfft(samples, **kwargs)
    freq = np.fft.rfftfreq(samples.shape[-1], 1/sample_rate)

    spi = sp.imag

    spi_abs = np.abs(spi)

    spi_abs = spi_abs / spi_abs.max()

    return freq, spi_abs


def features_fft2_cwt(samples, sample_rate, **kwargs):
    # we project the original wave to freq domain, then use max values
    # of freq domain as features.

    # takes     samples: np_array,  sample_rate: int
    # returns   features: np_array
    freq, spi2 = normed_fft2(samples, sample_rate)
    peakind = signal.find_peaks_cwt(spi2, np.arange(1, 10), **kwargs)
    # return freq[peakind], spi2[peakind], freq, spi2
    return freq[peakind]


def features_fft2_max(samples, sample_rate, **kwargs):
    # for max algo, it is suggested to specify a minimum prominance in kwargs
    # so that the noise peaks are not included
    # after some testing i think we can simply use 0.5? whatever
    freq, spi2 = normed_fft2(samples, sample_rate)
    peakind, _ = signal.find_peaks(spi2, **kwargs)
    # return freq[peakind], spi2[peakind], freq, spi2
    return freq[peakind]


def features_abs_fft_cwt(samples, sample_rate, **kwargs):
    # we project the original wave to freq domain, then use max values
    # of freq domain as features.

    # takes     samples: np_array,  sample_rate: int
    # returns   features: np_array
    freq, spi_abs = normed_abs_fft(samples, sample_rate)
    peakind = signal.find_peaks_cwt(spi_abs, np.arange(10, 20), **kwargs)
    # return freq[peakind], spi_abs[peakind], freq, spi_abs
    return freq[peakind]


def features_abs_fft_max(samples, sample_rate, **kwargs):
    # for max algo, it is suggested to specify a minimum prominance in kwargs
    # so that the noise peaks are not included
    # after some testing i think we can simply use 0.3? whatever
    freq, spi_abs = normed_abs_fft(samples, sample_rate)
    peakind, _ = signal.find_peaks(spi_abs, **kwargs)
    # return freq[peakind], spi_abs[peakind], freq, spi_abs
    return freq[peakind]


def features_seg_abs_fft_max(samples, sample_rate, **kwargs):
    # well, the fft method didn't work well for long pieces of complex music
    # like Piano Sonata No.11 (a 30 seconds slice of the music generates a lot
    # of noises, and the peak shapes are not good as well)
    # So i think actually we can use the spectrogram to estimate frequencies.
    # the spectrum is generated by fft over all segments and is thus 2 dimensional
    # we have already calculated spectrograms before (spectrogram.py), it directly
    # uses the spectrogram function from scipy
    # the function also takes these kwargs: nperseg=nperseg, nfft=nfft, noverlap=noverlap
    frequencies, times, spectrogram = signal.spectrogram(
        samples, fs=sample_rate, **kwargs)

    # for testing purposes, we can draw all spectrums of different times
    # # we may also need to normalize the spectrogram, but lets forget this for now
    # spectrogram = spectrogram / spectrogram.max()
    seg_spi = np.zeros(len(frequencies))
    # iterate over times to extract max
    for time_index, _ in enumerate(times):
        seg_spi = seg_spi + spectrogram[:, time_index]
    peakind, _ = signal.find_peaks(seg_spi)
    return frequencies[peakind], seg_spi[peakind], frequencies, seg_spi
    # return frequencies[peakind]


def features_seg_abs_fft_max_trunc(samples, sample_rate, **kwargs):
    first_n = 50
    freqs, intensities, a, b = features_seg_abs_fft_max(samples, sample_rate, **kwargs)
    first_inds = np.argsort(intensities)[len(intensities)-first_n:]
    return freqs[first_inds], intensities[first_inds], a, b


def features_seg_abs_fft_max_trunc_ratio(samples, sample_rate, **kwargs):
    features, _, _, _ = features_seg_abs_fft_max_trunc(samples, sample_rate, **kwargs)
    features.sort()
    return np.array([features[i]/features[i-1] for i in range(1, len(features))])


# temps = ['Just_Intonation', 'Pythagorean', 'Twelve_Tone_Equal']

# for temp in temps:
#     sample_rate, samples = wavfile.read('wavfiles/'+temp+'/mzt_331_3.wav.30.wav')
#     # a,b,c,d = features_abs_fft_max(samples, sample_rate, prominence=0.3 )
#     # a, b, c, d = features_abs_fft_cwt(samples, sample_rate, min_snr=5)
#     a, b, c, d = features_seg_abs_fft_max_trunc(samples, sample_rate, nperseg=4096, nfft=32768)
#     plt.plot(a, b, 'x')
#     plt.plot(c, d)

# e = features_seg_abs_fft_max_trunc_ratio(samples, sample_rate, nperseg=4096, nfft=32768)
# print(e)

# plt.show()
