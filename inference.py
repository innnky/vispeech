import logging
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)
logging.getLogger("jieba").setLevel(logging.INFO)
import matplotlib.pyplot as plt
import IPython.display as ipd
from mel_processing import spectrogram_torch
import os

import torch
from torch.utils.data import DataLoader
from utils import load_wav_to_torch

import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text.cleaner import text_to_sequence

# logging.getLogger("matplotlib").setLevel(logging.INFO)
# logging.getLogger("matplotlib").setLevel(logging.INFO)
def get_text(text):
    text_norm = text_to_sequence(text)
    text_norm = torch.LongTensor(text_norm)
    return text_norm
hps = utils.get_hparams_from_file("./configs/config.json")
net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.data.hop_length,
        hps.data.sampling_rate,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
_ = net_g.eval()

_ = utils.load_checkpoint("/Volumes/Extend/下载/G_76000 (1).pth", net_g, None)
text1 = "。下面给大家简单介绍一下怎么使用这个教程吧！首先我们要有魔法，才能访问到谷歌的云平台。点击连接并更改运行时类型，设置硬件加速器为G P U。然后，我们再从头到尾挨个点击每个代码块的运行标志。可能需要等待一定的时间。当我们进行到语音合成部分时，就可以更改要说的文本，并设置保存的文件名啦。"
# text2 = "。下面给大家简单介绍一下怎么使用这个教程吧！首先我们要有魔法，才能访问到谷歌的云平台。点击连接并更改运行时类型，设置硬件加速器为G P U。然后，我们再从头到尾挨个点击每个代码块的运行标志。并设置保存的文件名啦。"
stn_tst = get_text(text1)
with torch.no_grad():
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
    sid = torch.LongTensor([63])
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667)[0][0,0].data.cpu().float().numpy()
# ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
print(audio.shape[0]//44100)