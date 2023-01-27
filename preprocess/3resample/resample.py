
import os

import librosa
import soundfile
import tqdm
for spk in os.listdir("dataset/"):
    if os.path.isdir(f"dataset/{spk}"):
        if spk not in ["paimon_extend", "VO_paimon"]:
            continue
        for wavname in tqdm.tqdm(os.listdir(f"dataset/{spk}")):
            if wavname.endswith("wav"):
                print(wavname)
                wav, sr = librosa.load(f"dataset/{spk}/{wavname}", sr=22050)
                soundfile.write(f"dataset/{spk}/{wavname}", wav, sr)
