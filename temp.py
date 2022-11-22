import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import parselmouth
import librosa

fs = 22050
hop = 256
iii = 0
st = 4
pic = 3


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
    print( lll- len(wav_data)//hop_length)
    f0 = np.pad(f0, [[lpad, rpad]], mode="constant")


    return f0

outfile = open("filelists/paimon_extend.txt" ,"w")
for line in open("filelists/paimon_extend.txt.preprocessd").readlines():
    wav_path, phones, durations, sid = line.strip().split("|")
    phones = phones.split(" ")

    durations = [int(i) for i in durations.split(" ")]
    pitch = get_pitch(wav_path, sum(durations))
    nonzero_ids = np.where(pitch != 0)[0]
    try:
        interp_fn = interp1d(
            nonzero_ids,
            pitch[nonzero_ids],
            fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
            bounds_error=False,
        )
    except:
        continue
    pitch = interp_fn(np.arange(0, len(pitch)))
    pos = 0
    for i, d in enumerate(durations):
        if d > 0:
            pitch[i] = np.mean(pitch[pos: pos + d])
        else:
            pitch[i] = 0
        pos += d
    pitch = pitch[: len(durations)]
    nphf0 = " ".join(['{:.3f}'.format(i) for i in pitch])
    nphones = " ".join(phones)
    ndurations = " ".join([str(i) for i in durations])
    print(iii)
    outfile.write(f"{wav_path}|{nphones}|{ndurations}|{sid}|{nphf0}\n")
    #
    # if iii >st:
    #     print(wav_path)
    #     for phhh, f000 in zip(phones, phf0old):
    #         print( phhh, f000)
    #     plt.plot(phf0)
    #     plt.plot(phf0old)
    #     plt.plot(pitch)
    #     plt.show()
    #
    iii += 1
    # if iii >st+pic:
    #     break

outfile.close()