import logging

import soundfile
import torch
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write
import pydub
from flask import Flask, request, send_file, Response
import threading
app = Flask(__name__)
logging.getLogger("pydub").setLevel(logging.WARNING)
# 最大并发数 2
semaphore = threading.Semaphore(1)
def get_text(text):

    text_norm = text_to_sequence(text+"。")
    text_norm = torch.LongTensor(text_norm)
    return text_norm
hps = utils.get_hparams_from_file("configs/multispeaker.json")
net_g = SynthesizerTrn(
  len(symbols),
  hps.data.filter_length // 2 + 1,
  hps.data.hop_length,
  hps.data.sampling_rate,
  hps.train.segment_size // hps.data.hop_length,
  n_speakers=hps.data.n_speakers,
  **hps.model)

_ = net_g.eval()
_ = utils.load_checkpoint("ckpts/paimon.pth", net_g, None)
import time
import numpy as np

# def convert2mp3(wav, sr):
#     data = np.int16(wav * 2 ** 7)
#     song = pydub.AudioSegment(data.tobytes(), frame_rate=sr, sample_width=1, channels=1)
#     # return song.export(None, format="mp3", bitrate="320k")
#     return song.export(None, format="wav")

def tts(txt):
    res = None
    if semaphore.acquire(blocking=False):
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
                soundfile.write("temp.wav", audio, 22050)
                res = "temp.wav"
                # res = convert2mp3(audio, 22050)
                print("推理时间：", (t2 - t1), "s")
        finally:
            semaphore.release()
    return res

@app.route('/tts')
def text_api():
    text = request.args.get('text','')
    audio = tts(text)
    if audio is None:
        return "服务器忙"
    return send_file(audio, mimetype='audio/wav')

if __name__ == '__main__':
   app.run("0.0.0.0", 8080)
