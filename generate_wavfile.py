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
    # chuncate to fit length
    wave = wave[:int(end * sample_rate) - int(start * sample_rate)]
    samples[int(start * sample_rate):int(end * sample_rate)] = wave

    return samples


def mid_to_samples(mid, temperament, sample_rate):
    pitches = temperament()
    total_duration = mid.length
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
                # rightpad or chuncate generated wave so that we can safely add
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


# midfile_path = 'midfiles/mz_331_3_format0.mid'
# mid = mido.MidiFile(midfile_path)
# sample_rate = 44100
# samples = mid_to_samples(mid, temperament.twelve_tone_equal, sample_rate)
# wavfile.write('test_wavfiles/test2.wav', sample_rate, samples)

# --- generate Mozarts's Piano Sonata No.11 3 for all temeraments, save them to
# test_wavfiles ---
midfile_path = 'midfiles/mz_331_3_format0.mid'
mid = mido.MidiFile(midfile_path)
sample_rate = 44100
# temperaments = [temperament.Just_intonation, temperament.Pythagorean, temperament.twelve_tone_equal]
temp = temperament.Just_intonation
samples = mid_to_samples(mid, temp, sample_rate)
wavfile.write('test_wavfiles/Just_Intonation/mzt_331_3.wav',
              sample_rate, samples)

temp = temperament.Pythagorean
samples = mid_to_samples(mid, temp, sample_rate)
wavfile.write('test_wavfiles/Pythagorean/mzt_331_3.wav',
              sample_rate, samples)

temp = temperament.twelve_tone_equal
samples = mid_to_samples(mid, temp, sample_rate)
wavfile.write('test_wavfiles/Twelve_Tone_Equal/mzt_331_3.wav',
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
