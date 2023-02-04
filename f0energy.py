import os.path
import pathlib
import shutil

import librosa
import numpy as np
import parselmouth
from scipy.interpolate import interp1d


def stft(y):
    return librosa.stft(
        y=y,
        n_fft=1280,
        hop_length=512,
        win_length=1280,
    )

def rawenergy(y):
    # Extract energy
    S = librosa.magphase(stft(y))[0]
    e = np.sqrt(np.sum(S ** 2, axis=0))  # np.linalg.norm(S, axis=0)
    return e.squeeze()  # (Number of frames) => (654,)

def get_energy(path, p_len=None):
    wav, sr = librosa.load(path, 44100)
    e = rawenergy(wav)
    if p_len is None:
        p_len = wav.shape[0] // 512
    assert e.shape[0] -p_len <2 ,(e.shape[0] ,p_len)
    e = e[: p_len]
    return e


def get_pitch(path,lll):
    """
    :param wav_data: [T]
    :param mel: [T, 80]
    :param config:
    :return:
    """
    fs = 44100
    hop = 512
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
    assert 0<=rpad<=2,(len(f0), lll, len(wav_data)//hop_length)
    assert 0<=( lll- len(wav_data)//hop_length)<=1
    f0 = np.pad(f0, [[lpad, rpad]], mode="constant")

    return f0

iii = 0
with open(f"filelists/train.txt" ,"w") as outfile:
    for line in open(f"filelists/files.dur").readlines():
        spk,id_, phones, durations = line.strip().split("|")
        pathlib.Path(f"dataset/{spk}").mkdir(exist_ok=True)
        wav_path = f"mfa_temp/wavs/{spk}/{id_}.wav"
        target_path = f"dataset/{spk}/{id_}.wav"
        phones = phones.split(" ")

        durations = [int(i) for i in durations.split(" ")]
        try:
            pitch = get_pitch(wav_path, sum(durations))
        except:
            continue
        # np.save(target_path+".f0.npy", pitch)
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

        energy = get_energy(wav_path, sum(durations))
        pos = 0
        for i, d in enumerate(durations):
            if d > 0:
                energy[i] = np.mean(energy[pos: pos + d])
            else:
                energy[i] = 0
            pos += d
        energy = energy[: len(durations)]
        phenergy = " ".join(['{:.3f}'.format(i) for i in energy])

        phones = " ".join(phones)
        durations = " ".join([str(i) for i in durations])
        shutil.move(wav_path, target_path)
        print(iii, wav_path)
        outfile.write(f"{spk}|{id_}|{phones}|{durations}|{nphf0}|{phenergy}\n")
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

