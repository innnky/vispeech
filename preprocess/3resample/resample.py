
import os

import librosa
import soundfile

for spk in os.listdir("dataset/"):
    if os.path.isdir(f"dataset/{spk}"):
        for wavname in os.listdir(f"dataset/{spk}"):
            if wavname.endswith("wav"):
                wav, sr = librosa.load(f"dataset/{spk}/{wavname}")
                soundfile.write(f"dataset/{spk}/{wavname}", wav, sr)
