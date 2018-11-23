
import numpy as np

def twelve_tone_equal(A4=69, n_notes=127, pitch_A4=440.00):
    # 12-tone equal temporament
    # pitches[69] is the 69th note
    pitches = np.zeros(n_notes)
    pitches[A4] = pitch_A4

    for i in range(len(pitches)):
        pitches[i] = pitches[A4] * 2**((i - 69) / 12)

    return pitches
