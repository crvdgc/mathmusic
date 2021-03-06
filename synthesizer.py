from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import mido


import temperament


def genwave(duration=1.0, start=0.0, end=1.0, vel=1.0, freq=440.00, sample_rate=44100):
    # generate a segmented wave, like this  -----------/\/\/\/\/\/\----------
    #                                       |<---------duration------------>|
    #                                       |          ↑start     ↑end      |
    # the param vel is velocity, which means the amplitude of this wave

    # generate a slightly longer wave
    sample_points = np.arange(0, end - start + 0.1, 1/sample_rate)
    wave = np.sin(freq * sample_points * 2 * np.pi) * vel

    samples = np.zeros(int(duration * sample_rate))
    # truncate to fit length
    wave = wave[:int(end * sample_rate) - int(start * sample_rate)]
    try:
        samples[int(start * sample_rate):int(end * sample_rate)] = wave
    except ValueError as e:
        print("ValueError")
        print(e.args)
        print(duration, start, end, vel, freq, sample_rate)
        print(wave)
        print(samples)

    return samples


# hz_freqs = dict()
# for freq in range(2000):
#     hz_freqs[freq] = genwave(freq=freq)

# def writeHz(freqs):
#     res = np.zeros(len(hz_freqs[0]))
#     for i in freqs:
#         res = res + hz_freqs[i]
#     wavfile.write('+'.join(map(str, freqs)) + '.wav', 44100, res)
#     return res

# def writeHzVel(freqs, vels):
#     res = np.zeros(len(hz_freqs[0]))
#     for i in range(len(freqs)):
#         res = res + hz_freqs[freqs[i]] * vels[i]
#     wavfile.write('+'.join(map(str, freqs)) + '.wav', 44100, res)
#     return res

def genwave_mut(samples, duration=1.0, start=0.0, end=1.0, vel=1.0, freq=440.00, sample_rate=44100):
    # mutating function version of genwave, avoid too many mallocs by always using the
    # same array to store generated wave
    # generate a segmented wave, like this  -----------/\/\/\/\/\/\----------
    #                                       |<---------duration------------>|
    #                                       |          ↑start     ↑end      |
    # the param vel is velocity, which means the amplitude of this wave

    # generate a slightly longer wave
    sample_points = np.arange(0, end - start + 0.1, 1/sample_rate)
    wave = np.sin(freq * sample_points * 2 * np.pi) * vel
    # clear our wave
    samples[:] = 0 * samples
    # truncate to fit length
    wave = wave[:int(end * sample_rate) - int(start * sample_rate)]
    try:
        samples[int(start * sample_rate):int(end * sample_rate)] = wave
    except ValueError as e:
        print("ValueError")
        print(e.args)
        print(duration, start, end, vel, freq, sample_rate)
        print(wave)
        print(samples)

    # we don't need to return it, but we still do this
    return samples


def mid_to_samples(mid, temperament, sample_rate):
    pitches = temperament()
    # this duration reported by mid.length cannot be fully trusted,
    # test shows that this value can shift up to 0.2s for a simple 190s song
    # so we multiply it by a factor of 1.001 to avoid truncating at end of song.
    total_duration = mid.length * 1.001
    samples = np.zeros(int(total_duration * sample_rate) + 1)

    for i, track in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, track.name))
        # we need to keep a reference of current time when iterating over msgs.
        # unit: s
        current_time = 0

        # the default tempo is 500000 (ms / beat)
        current_tempo = 500000

        # there are 3 conditions when a note is ended:
        # 1. we have recieved note_off event
        # 2. we have recieved note_on event, but velocity is 0
        # 3. a new note started, so this note must end
        # when a note is ended, we must then look up the time and the velocity when
        # it started.
        # so we also need to keep a dict that stores such information
        note_on_dict = dict()

        for j, msg in enumerate(track):
            # if j > 1000:
            #     break
            print(j, msg, current_time)
            # first, the time attr of each msg is the time since last msg,
            # so we need to accumulate this value (this is the absolute start time)
            # msg.time is in ticks, so we need to convert it to seconds
            if msg.type == 'set_tempo':
                current_tempo = msg.tempo
            time_delta = mido.tick2second(
                msg.time, mid.ticks_per_beat, current_tempo)
            current_time = current_time + time_delta

            # then we check if this is the end of a note
            if ((msg.type == 'note_off') or (msg.type == 'note_on' and (msg.velocity == 0 or msg.note in note_on_dict))) and msg.note in note_on_dict:
                # ends
                start, vel = note_on_dict.pop(msg.note)
                end = current_time
                # patch the wave
                wave_patch = genwave(
                    duration=total_duration, start=start, end=end, vel=vel, freq=pitches[msg.note])
                # rightpad or truncate generated wave so that we can safely add
                if len(wave_patch) > len(samples):
                    wave_patch = wave_patch[:len(samples)]
                elif len(wave_patch) < len(samples):
                    wave_patch = np.pad(
                        wave_patch, (0, len(samples) - len(wave_patch)), 'constant')
                else:
                    pass
                # add to original wave
                samples = samples + wave_patch
            # if this is note a end of a note, i.e. this is a new note pushed, then we simply add
            # it to our dict
            elif (msg.type == 'note_on'):
                note_on_dict[msg.note] = (current_time, msg.velocity)
            # finally, if this message is irrelavent
            else:
                pass
    return samples


def mid_to_samples_mut(mid, temperament, sample_rate, ignore_vel=False):
    # use genwave_mut instead of genwave version of mid_to_samples
    pitches = temperament()
    # this duration reported by mid.length cannot be fully trusted,
    # test shows that this value can shift up to 0.2s for a simple 190s song
    # so we multiply it by a factor of 1.001 to avoid truncating at end of song.
    total_duration = mid.length * 1.001
    samples = np.zeros(int(total_duration * sample_rate) + 1)
    wave_patch = np.zeros(len(samples))

    for i, track in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, track.name))
        # we need to keep a reference of current time when iterating over msgs.
        # unit: s
        current_time = 0

        # the default tempo is 500000 (ms / beat)
        current_tempo = 500000

        # there are 3 conditions when a note is ended:
        # 1. we have recieved note_off event
        # 2. we have recieved note_on event, but velocity is 0
        # 3. a new note started, so this note must end
        # when a note is ended, we must then look up the time and the velocity when
        # it started.
        # so we also need to keep a dict that stores such information
        note_on_dict = dict()

        for j, msg in enumerate(track):
            # if j > 100:
            #     break
            print(j, msg, current_time)
            # first, the time attr of each msg is the time since last msg,
            # so we need to accumulate this value (this is the absolute start time)
            # msg.time is in ticks, so we need to convert it to seconds
            if msg.type == 'set_tempo':
                current_tempo = msg.tempo
            time_delta = mido.tick2second(
                msg.time, mid.ticks_per_beat, current_tempo)
            current_time = current_time + time_delta

            # then we check if this is the end of a note
            if ((msg.type == 'note_off') or (msg.type == 'note_on' and (msg.velocity == 0 or msg.note in note_on_dict))) and msg.note in note_on_dict:
                # ends
                start, vel = note_on_dict.pop(msg.note)
                end = current_time
                genwave_mut(wave_patch,
                            duration=total_duration, start=start, end=end, vel=vel, freq=pitches[msg.note])
                # add to original wave
                samples = samples + wave_patch
            # if this is note a end of a note, i.e. this is a new note pushed, then we simply add
            # it to our dict
            elif (msg.type == 'note_on'):
                note_on_dict[msg.note] = (current_time, msg.velocity)
            # finally, if this message is irrelavent
            else:
                pass
    return samples


# midfile_path = 'midfiles/mz_545_3_format0.mid'
# mid = mido.MidiFile(midfile_path)
# sample_rate = 44100
# samples = mid_to_samples_mut(
#     mid, temperament.twelve_tone_equal, sample_rate, ignore_vel=True)
# wavfile.write('test_wavfiles/test.wav', sample_rate, samples)
# plt.plot(samples[:441000])
# plt.show()

# --- generate Mozarts's Piano Sonata No.11 3 for all temeraments, save them to
# test_wavfiles ---
midfile_path = 'midfiles/mz_330_3_format0.mid'
mid = mido.MidiFile(midfile_path)
sample_rate = 44100
# temperaments = [temperament.Just_intonation, temperament.Pythagorean, temperament.twelve_tone_equal]
temp = temperament.Just_intonation
samples = mid_to_samples_mut(mid, temp, sample_rate)
wavfile.write('test_wavfiles/Just_Intonation/mz_330_3.wav',
              sample_rate, samples)

temp = temperament.Pythagorean
samples = mid_to_samples_mut(mid, temp, sample_rate)
wavfile.write('test_wavfiles/Pythagorean/mz_330_3.wav',
              sample_rate, samples)

temp = temperament.twelve_tone_equal
samples = mid_to_samples_mut(mid, temp, sample_rate)
wavfile.write('test_wavfiles/Twelve_Tone_Equal/mz_330_3.wav',
              sample_rate, samples)


# --- sample 440Hz ---

# samples = genwave()
# sample_rate= 44100
# # save the wave file
# wavfile.write('wavfiles/test.wav', sample_rate, samples)


# # --- sample scale ---

# tonesscale = [0,2,4,5,7,9,11,12]

# import temperament
# pitches = temperament.Just_intonation()

# sample_rate = 44100

# samples = np.zeros(int(len(tonesscale) * sample_rate / 2))
# for i in range(len(tonesscale)):
#     note = tonesscale[i]
#     new_wave = genwave(duration=len(tonesscale)/2, start=i /
#                        2, end=(i+1)/2, freq=pitches[note+60])
#     samples = samples + new_wave

# # smooth the song to avoid noise between two note
# samples = pd.DataFrame(samples)
# samples = samples.rolling(2).mean()
# samples = np.array(samples)[2:]

# wavfile.write('wavfiles/scale_Just_intonation.wav', sample_rate, samples)

# # --- simple song ---

# a_simple_song = [4, 4, 9, 11, 12, 11, 9, 9,
#                  5, 5, 4, 2, 4, 4, 4, 4,
#                  4, 4, 9, 11, 12, 11, 9, 9,
#                  5, 2, 4, 4, -3, -3, -3, -3]

# from temperament import twelve_tone_equal
# pitches = twelve_tone_equal()

# sample_rate = 44100

# samples = np.zeros(int(len(a_simple_song) * sample_rate / 2))
# for i in range(len(a_simple_song)):
#     note = a_simple_song[i]
#     new_wave = genwave(duration=len(a_simple_song)/2, start=i /
#                        2, end=(i+1)/2, freq=pitches[note+60-12])
#     samples = samples + new_wave

# # smooth the song to avoid noise between two note
# samples = pd.DataFrame(samples)
# samples = samples.rolling(2).mean()
# samples = np.array(samples)[2:]

# wavfile.write('wavfiles/simple_song.wav', sample_rate, samples)
