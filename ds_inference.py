import re

import soundfile
import torch

from modules import commons
from utils import utils
from egs.visinger2.models import SynthesizerTrn
from text.cleaner import text_to_sequence

speaker = "biaobei"
config_json = "egs/visinger2/config.json"
checkpoint_path = f"/Volumes/Extend/下载/G_60000.pth"
step = re.findall(r'G_(\d+)\.pth', checkpoint_path)[0]
text = "[JA]私が思う標準貝は日本語も話せる。[JA]"
text = "为了保护尤摩扬人民不受异虫的残害，我所做的，比他们自己的领导委员会都多。"

hps = utils.get_hparams_from_file(config_json)
net_g = SynthesizerTrn(hps)
_ = net_g.eval()
_ = utils.load_checkpoint(checkpoint_path, net_g, None)

def infer(model, hps, text, speaker):

    spkid = hps.data.spk2id[speaker]
    phseq = text_to_sequence(text)
    phseq = commons.intersperse(phseq, 0)

    text_norm = torch.LongTensor(phseq)
    x_tst = text_norm.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([text_norm.size(0)])
    spk = torch.LongTensor([spkid])
    with torch.no_grad():
        infer_res = model.infer(x_tst, x_tst_lengths, None, None, None, gtdur=None,
                                                        spk_id=spk)
    seg_audio = infer_res[0][0, 0].data.float().numpy()
    return seg_audio


audio = infer(net_g, hps, text, speaker)
soundfile.write(f"samples/{speaker}_{text[:5]}_{step}step.wav", audio, 44100)
