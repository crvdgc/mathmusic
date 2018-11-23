import scipy.io.wavfile as wav
from python_speech_features import mfcc, delta, logfbank
import numpy as np


def mfcc_features(filename: str):
    (rate, sig) = wav.read(filename)
    mfcc_feat = mfcc(sig, rate)
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig, rate)
    return np.concatenate((mfcc_feat, d_mfcc_feat, fbank_feat), axis=1)
