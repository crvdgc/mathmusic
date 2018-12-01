import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
from matplotlib.ticker import MultipleLocator
from datetime import datetime
import sys

from spectrogram import draw_log_scale_spectrogram_12tone

# params: nperseg nfft noverlap filename
nperseg = int(sys.argv[1])
nfft = int(sys.argv[2])
noverlap = int(sys.argv[3])

wavfile_name = sys.argv[4]


def escape_for_filename(filename):
    return filename.replace('/', '_').replace('.', '_')


sample_rate, samples = wavfile.read(wavfile_name)


print('Drawing for {filename} with params: nperseg={nperseg}, nfft={nfft}, noverlap={noverlap}'.format(
    filename=wavfile_name, nperseg=nperseg, nfft=nfft, noverlap=noverlap))

start_time = datetime.now()

try:
    draw_log_scale_spectrogram_12tone(samples, sample_rate, nperseg=nperseg, nfft=nfft,
                                      noverlap=noverlap, vmax=None, dpi=300, cmap='nipy_spectral', filename=escape_for_filename(wavfile_name))
except TimeoutError as e:
    print("Timed out!")
except KeyboardInterrupt as e:
    print(e)
    exit(0)
except Exception as inst:
    print('Unexpected Exception happened: ', inst)


time_elapsed = datetime.now() - start_time

print('Done. Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
