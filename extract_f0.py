filelist = "filelists/traintaffy.txt"
# filelists/val.txt
import librosa
import pyworld
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import parselmouth
from tqdm import tqdm

fs = 44100
hop = 512
#
def resize2d(x, target_len):
    source = np.array(x)
    source[source<0.001] = np.nan
    target = np.interp(np.arange(0, len(source)*target_len, len(source))/ target_len, np.arange(0, len(source)), source)
    res = np.nan_to_num(target)
    return res

def compute_f0(path, c_len):
    x, sr = librosa.load(path, sr=fs)
    f0, t = pyworld.dio(
        x.astype(np.double),
        fs=sr,
        f0_ceil=800,
        frame_period=1000 * hop / sr,
    )
    f0 = pyworld.stonemask(x.astype(np.double), f0, t, fs)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    assert abs(c_len - x.shape[0]//hop) < 3, (c_len, f0.shape)

    return None, resize2d(f0, c_len)
import os
with open(filelist) as f:
    for line in tqdm(f.readlines()):
        line = line.strip()
        wavpath, raw_phones, raw_durations, spk = line.split("|")[0], line.split("|")[1], line.split("|")[2], line.split("|")[3]
        durations = [int(i) for i in raw_durations.split(" ")]
        if not os.path.exists(wavpath+".f0.npy"):
            _, f0 = compute_f0(wavpath, sum(durations))
            # print(wavpath)
            np.save(wavpath+".f0.npy", f0)

