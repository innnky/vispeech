import re
import jieba.posseg as psg
import os
import pathlib
import shutil
from preprocess.config import spk, transcription_path
# from paddlespeech.t2s.frontend.zh_frontend import Frontend

# frontend = Frontend()
from text import text_to_phones

spk_name = spk
pathlib.Path(f"preprocess/1mfa/mfa_dataset").mkdir(exist_ok=True)
ttsdict = {line.strip().split(' ')[0]:line.strip().split(' ')[1:]  for line in open("preprocess/dict/simple.lexicon").readlines()}
all_pinyin = ttsdict.keys()
for line in open(transcription_path).readlines():
    path, txt  = line.strip().split('|')[0:2]
    filename = path.split("/")[-1]
    pathlib.Path(f"preprocess/1mfa/mfa_dataset/{spk_name}").mkdir(exist_ok=True)
    if os.path.exists(path):
        shutil.copy(path, f"preprocess/1mfa/mfa_dataset/{spk_name}/{filename}")

    if os.path.exists(f"preprocess/1mfa/mfa_dataset/{spk_name}/{filename}"):
        pinyins = text_to_phones(txt)
        # print(pinyins)
        if len(pinyins) == 0:
            print(txt)
        label_path = f"preprocess/1mfa/mfa_dataset/{spk_name}/{filename}".replace("wav", "lab")

        with open(label_path, "w") as o:
            o.write(" ".join(pinyins)+"\n")
        # print(path)
