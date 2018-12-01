from scipy.io import wavfile

#@TODO
def seperate_channels(samples):
    pass


def slice_wavfile(wavfile_name, time_interval=30):
    sample_rate, samples = wavfile.read(wavfile_name)
    music_length = len(samples) // sample_rate + 1
    for i in range(0, music_length, time_interval):
        if (i+time_interval)*sample_rate <= len(samples):
            samples_slice = samples[i*sample_rate:(i+time_interval)*sample_rate]
        else:
            samples_slice = samples[i*sample_rate:]
        wavfile.write(wavfile_name + '.' + str(i) + '.wav', sample_rate, samples_slice)

slice_wavfile('wavfiles/test2.wav')