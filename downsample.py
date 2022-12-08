import librosa
import os
import soundfile
rootdir = "/Volumes/Extend/下载/vispeech-mb备份/vctk"
i = 0

for spk in sorted(os.listdir(rootdir)):
    if os.path.isdir(f"{rootdir}/{spk}"):
        for filename in sorted(os.listdir(f"{rootdir}/{spk}")):
            if filename.endswith("wav"):
                wav, sr = librosa.load(f"{rootdir}/{spk}/{filename}", 22050)
                soundfile.write(f"{rootdir}/{spk}/{filename}", wav, sr)
                i += 1
                print(i)