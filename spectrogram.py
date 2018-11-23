import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
from matplotlib.ticker import MultipleLocator
from datetime import datetime
import timeout_decorator


@timeout_decorator.timeout(300)
def draw_log_scale_spectrogram_12tone(samples, sample_rate, nperseg=4096, nfft=4096, noverlap=None, vmax=None, dpi=300, cmap='gray', filename='spectrogram'):
    # y-axis is MIDI note number according to 12tone equal temperament

    print(datetime.now(), "Calculating spectrogram...")
    frequencies, times, spectrogram = signal.spectrogram(
        samples, fs=sample_rate, nperseg=nperseg, nfft=nfft, noverlap=noverlap)
    notes = np.log2(frequencies/440) * 12 + 69
    # spectrogram = np.log(spectrogram + 1)
    # remove -inf
    notes[0] = -999

    print(datetime.now(), "Plotting colormesh...")
    # only include common frequencies
    ymin = 40
    ymax = 100
    fig, ax = plt.subplots(figsize=(80, 10))
    plt.axis([times.min(), times.max(), ymin, ymax])

    plt.pcolormesh(times, notes, spectrogram, vmin=0, vmax=vmax, cmap=cmap)
    # plt.yticks(np.arange(ymin, ymax, 1))
    minorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    plt.ylabel('MIDI Note [log2 Hz]')
    plt.xlabel('Time [second]')
    plt.colorbar()

    print(datetime.now(), "Writing to file ...")
    plt.savefig('figs/{filename}_log_freq_log_amp_nperseg{nperseg}_nfft{nfft}_noverlap{noverlap}.png'.format(
        filename=filename, nperseg=nperseg, nfft=nfft, noverlap=noverlap), dpi=dpi, bbox_inches='tight')
    plt.close(fig='all')
