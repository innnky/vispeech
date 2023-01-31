import glob
import os
import sys
import argparse
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils import audio
import utils.utils as utils
from tqdm import tqdm
import pyworld as pw
from random import shuffle
from scipy.stats import betabinom

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("numba").setLevel(logging.INFO)

def extract_mel(wav, hparams):
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    return mel_spectrogram.T, wav

def extract_pitch(wav, hps):
    # rapt may be better
    f0, _ = pw.harvest(wav.astype(np.float64),
                   hps.sample_rate,
                   frame_period=hps.hop_size / hps.sample_rate * 1000)
    return f0

def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=1.0):
    P, M = phoneme_count, mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling_factor*i, scaling_factor*(M+1-i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return np.array(mel_text_probs)


def extract_attn_prior(mels, text):
    beta_binomial_scaling_factor = 1.0
    # Calculate attention prior
    attn_prior = beta_binomial_prior_distribution(
        mels.shape[0],
        len(text)*2+1,
        beta_binomial_scaling_factor,
    )
    return attn_prior


def process_utterance(hps, data_root, item, id2text):
    out_dir = data_root
    text = [i for i in id2text[item].split(" ")]
    wav_path = os.path.join(data_root, "wavs",
                            "{}.wav".format(item))
    wav = audio.load_wav(wav_path,
                         raw_sr=hps.data.sample_rate,
                         target_sr=hps.data.sample_rate,
                         win_size=hps.data.win_size,
                         hop_size=hps.data.hop_size)

    mel, _ = extract_mel(wav, hps.data)
    out_mel_dir = os.path.join(out_dir, "mels")
    os.makedirs(out_mel_dir, exist_ok=True)
    mel_path = os.path.join(out_mel_dir, item)
    np.save(mel_path, mel)

    pitch = extract_pitch(wav, hps.data)
    out_pitch_dir = os.path.join(out_dir, "pitch")
    os.makedirs(out_pitch_dir, exist_ok=True)
    pitch_path = os.path.join(out_pitch_dir, item)
    np.save(pitch_path, pitch)

    attn_prior = extract_attn_prior(mel, text)
    out_attn_prior_dir = os.path.join(out_dir, "attn_prior")
    os.makedirs(out_attn_prior_dir, exist_ok=True)
    attn_prior_path = os.path.join(out_attn_prior_dir, item)
    np.save(attn_prior_path, attn_prior)



def process(args, hps, data_dir):
    print(os.path.join(data_dir, "wavs"))
    transcriptions = open(os.path.join(data_dir, "transcriptions.txt")).readlines()
    id2text = {i.strip().split("|")[0]: i.strip().split("|")[2] for i in transcriptions}
    if(not os.path.exists(os.path.join(data_dir, "file.list"))):
        with open(os.path.join(data_dir, "file.list") , "w") as out_file:
            files = os.listdir(os.path.join(data_dir, "wavs"))
            files = [i for i in files if i.endswith(".wav") and i.replace(".wav", "") in id2text.keys()]
            for f in files:
                out_file.write(f.strip().split(".")[0] + '\n')
    metadata = [
        item.strip() for item in open(
            os.path.join(data_dir, "file.list")).readlines()
    ]
    executor = ProcessPoolExecutor(max_workers=args.num_workers)
    results = []
    for item in metadata:
        results.append(executor.submit(partial(process_utterance, hps, data_dir, item,id2text)))
    return [result.result() for result in tqdm(results)]

def split_dataset(data_dir):
    metadata = [
        item.strip() for item in open(
            os.path.join(data_dir, "file.list")).readlines()
    ]
    shuffle(metadata)
    train_set = metadata[:-2]
    test_set =  metadata[-2:]
    with open(os.path.join(data_dir, "train.list"), "w") as ts:
        for item in train_set:
            ts.write(item+"\n")
    with open(os.path.join(data_dir, "test.list"), "w") as ts:
        for item in test_set:
            ts.write(item+"\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='egs/visinger2/config.json',
                        help='json files for configurations.')
    parser.add_argument('--num_workers', type=int, default=int(cpu_count()))

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config)
    spklist = [spk for spk in os.listdir("data") if os.path.isdir(f"data/{spk}") and not os.path.exists(f"data/{spk}/test.list")]
    for spk in tqdm(spklist):
        print(f"preprocessing {spk}")
        data_dir = f"data/{spk}"
        process(args, hps, data_dir)
        split_dataset(data_dir)

if __name__ == "__main__":
    main()
