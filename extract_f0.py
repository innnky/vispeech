filelist = "filelists/paimon_extend.txt.preprocessd"
import librosa
import pyworld
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import parselmouth
fs = 22050
hop = 256
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
#

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

i = 0
#
outstr = ''
with open(filelist) as f:
    for line in f.readlines():
        line = line.strip()
        wavpath, raw_phones, raw_durations, spk = line.split("|")[0], line.split("|")[1], line.split("|")[2], line.split("|")[3]
        durations = [int(i) for i in raw_durations.split(" ")]
        phones = raw_phones.split(" ")
        f0 = get_pitch(wavpath, sum(durations))
        phf0s = []
        current_frame = 0
        for dur in durations:
            chunk = f0[current_frame:current_frame+dur]
            phf0s.append(np.average(chunk[chunk!=0]))
            # phf0s.append(np.average(chunk))
            current_frame+=dur
        assert abs(current_frame - len(f0)) <2
        phf0s = " ".join(['{:.3f}'.format(i) for i in phf0s])
        phf0s = phf0s.replace("nan", "0.000")
        outstr  += f"{wavpath}|{raw_phones}|{raw_durations}|{spk}|{phf0s}\n"
        print(wavpath, i, phf0s)
        i += 1
        np.save(wavpath+".f0.npy", f0)
with open(filelist+"newnew", "w") as f:
    f.write(outstr)


def normalize(f0paths, mean, std):
    max_value = np.finfo(np.float64).min
    min_value = np.finfo(np.float64).max
    for f0path in f0paths:
        values = (np.load(f0path) - mean) / std
        np.save(f0path, values)

        max_value = max(max_value, max(values))
        min_value = min(min_value, min(values))

    return min_value, max_value
#
# pitch_scaler = StandardScaler()
# with open(filelist) as f:
#     f0paths = []
#     for line in f.readlines():
#         wavpath = line.split("|")[0]
#         f0 = np.load(wavpath+".f0.npy")
#         f0paths.append(wavpath+".f0.npy")
#         pitch_scaler.partial_fit(f0.reshape((-1, 1)))
#     pitch_mean = pitch_scaler.mean_[0]
#     pitch_std = pitch_scaler.scale_[0]
#     print(pitch_mean, pitch_std)
#     min_value, max_value = normalize(f0paths, pitch_mean, pitch_std)
#     print(min_value, max_value)
# 标贝
# -1.2502422699374525 5.283679553058448
