
import numpy as np


def twelve_tone_equal(A4=69, n_notes=127, pitch_A4=440.00):
    # 12-tone equal temporament
    # pitches[69] is the 69th note
    #   0       1       2       3       4       5       6       7       8       9       10      11      12      (No.)
    #   C       #C      D       #D      E       F       #F      G       #G      A       #A      B       C       (Name, #)
    #   C       bD      D       bE      E       F       bG      G       bA      A       bB      B       C       (Name, b)
    #   do              re              mi      fa              sol             la              si      do      (C Major)
    #   60      61      62      63      64      65      66      67      68      69      70      71      72      (MIDI)
    #   1.0000  1.0595  1.1225  1.1892  1.2599  1.3348  1.4142  1.4983  1.5874  1.6818  1.7818  1.8878  2.0000
    #   2.0000  2.1189  2.2449  2.3784  2.5198  2.6697  2.8284  2.9966  3.1748  3.3636  3.5636  3.7755  4.0000
    #   4.0000  4.2379  4.4899  4.7568  5.0397  5.3394  5.6569  5.9932  6.3496  6.7272  7.1272  7.5510  8.0000
    #   8.0000  8.4757  8.9797  9.5137  10.079  10.679  11.314  11.987  12.699  13.454  14.254  15.102  16.000  (approx. freq.)

    pitches = np.zeros(n_notes)
    pitches[A4] = pitch_A4

    for i in range(len(pitches)):
        pitches[i] = pitches[A4] * 2**((i - 69) / 12)

    return pitches

def another_mod(n, m):
    if n > 0:
        return n%m
    elif n == 0:
        return 0
    else:
        return n%m - m

def Pythagorean(A4=69, n_notes=127, pitch_A4=440.00):
    # 五度相生律（双参数m,n）

    pitches = np.zeros(n_notes)
    pitches[A4] = pitch_A4

    octave = np.zeros(12)
    octave[0] = 1.0

    for i in range(1, len(octave)):
        octave[i] = octave[i-1] * 1.50
        if octave[i] > 2.00:
            octave[i] = octave[i] / 2.00

    octave = np.sort(octave)
    octave = octave * pitch_A4 / octave[9]

    pitches[69-9:69-9+12] = octave

    C4 = A4 - 9
    bias = C4 % 12
    for i in range(n_notes):
        pitches[i] = pitches[C4 + (i-bias) % 12] * 2 ** ( (i-bias) // 12 - C4 // 12 )

    return pitches
