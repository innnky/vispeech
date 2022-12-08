import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import io
from scipy.io.wavfile import write
import os
from flask import Flask, request, send_file
import threading
app = Flask(__name__)
mutex = threading.Lock()

def get_text(text):

    text_norm = text_to_sequence(text+"。")
    text_norm = torch.LongTensor(text_norm)
    return text_norm
hps = utils.get_hparams_from_file("configs/ms.json")
net_g = SynthesizerTrn(
  len(symbols),
  hps.data.filter_length // 2 + 1,
  hps.data.hop_length,
  hps.data.sampling_rate,
  hps.train.segment_size // hps.data.hop_length,
  n_speakers=hps.data.n_speakers,
  **hps.model)

_ = net_g.eval()

_ = utils.load_checkpoint("/Volumes/Extend/下载/temp.pth", net_g, None)
import time
import numpy as np
def tts(txt):
    audioname = None
    if mutex.acquire(blocking=False):
        try:
            stn_tst = get_text(txt)
            with torch.no_grad():
                x_tst = stn_tst.unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
                t1 = time.time()
                spk = torch.LongTensor([1])

                audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8,sid=spk,
                                            length_scale=1)[0][0, 0].data.float().numpy()
                t2 = time.time()
                audioname = "c.wav"
                write(audioname ,44100, audio)
                os.system("ffmpeg -i c.wav -ar 22050 -y converted.wav")
                audioname = "converted.wav"

                print("推理时间：", (t2 - t1), "s")
        finally:
            mutex.release()
    return audioname

@app.route('/tts')
def text_api():
    text = request.args.get('text','')
    audio = tts(text)
    if audio is None:
        return "服务器忙"
    return send_file(audio,as_attachment=True)


if __name__ == '__main__':
   app.run("0.0.0.0", 8080)
