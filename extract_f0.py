filelist = "filelists/energy_train(1).txt"
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
def get_pitch(path,lll):
    """
    :param wav_data: [T]
    :param mel: [T, 80]
    :param config:
    :return:
    """
    sampling_rate = fs
    hop_length = hop
    wav_data, _ = librosa.load(path,sampling_rate)
    time_step = hop_length / sampling_rate * 1000
    f0_min = 80
    f0_max = 750

    f0 = parselmouth.Sound(wav_data, sampling_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array["frequency"]
    lpad = 2
    rpad = lll - len(f0) - lpad
    assert 0<=rpad<=2
    f0 = np.pad(f0, [[lpad, rpad]], mode="constant")

    return f0

def compute_f0(path):
    x, sr = librosa.load(path, sr=fs)
    assert sr == fs
    f0, t = pyworld.dio(
        x.astype(np.double),
        fs=sr,
        f0_ceil=800,
        frame_period=1000 * hop / sr,
    )
    f0 = pyworld.stonemask(x.astype(np.double), f0, t, fs)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    return f0

with open(filelist) as f:
    for line in tqdm(f.readlines()):
        line = line.strip()
        wavpath, raw_phones, raw_durations, spk = line.split("|")[0], line.split("|")[1], line.split("|")[2], line.split("|")[3]
        durations = [int(i) for i in raw_durations.split(" ")]
        f0 = get_pitch(wavpath, sum(durations))
        # print(wavpath)
        np.save(wavpath+".f0.npy", f0)

