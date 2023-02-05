
import os
import random

import numpy as np
import tgt
import tqdm

from text.symbols import pu_symbols

silence_symbol = ["sil", "sp", "spn"]
sampling_rate = 44100
hop_length = 512

def sample(probabilities):
    probabilities = np.maximum(probabilities, 0)
    normalized_probs = probabilities / np.sum(probabilities)
    return np.random.choice(len(probabilities), p=normalized_probs)

def get_probability(x, minimum, maximum, mean):
    if x <= minimum or x >= maximum:
        return 0
    if x == mean:
        return 1
    if x < mean:
        return (x - minimum) / (mean - minimum)
    if x > mean:
        return (maximum - x) / (maximum - mean)



def get_sp(frames, is_last, is_first):
    if is_first:
        return "sp"
    if is_last:
        if random.random()<0.3:
            return "sp"
        else:
            return "."
    pu_dict={
        ",":[3,15,40],
        "…":[30,1000,1000]
    }
    probabilities = []
    for i in [",","…"]:
        probabilities.append(get_probability(frames, *pu_dict[i]))
    probabilities.append(0.01)
    return [",","…","sp"][sample(probabilities)]

def get_alignment(tier):
    phones = []
    durations = []
    end_time = []
    last_end = 0
    for t in tier._objects:
        start, end, phone = t.start_time, t.end_time, t.text
        # print(f"音素：{phone}，开始:{start}，结束:{end}")
        # Trim leading silences
        if last_end != start:
            durations.append(
                int(
                    np.round(start * sampling_rate / hop_length)
                    - np.round(last_end * sampling_rate / hop_length)
                )
            )
            phones.append('sp')
            end_time.append(start)

        phones.append(phone)
        durations.append(
            int(
                np.round(end * sampling_rate / hop_length)
                - np.round(start * sampling_rate / hop_length)
            )
        )
        end_time.append(end)

        last_end = end

    if tier.end_time != last_end:
        durations.append(
            int(
                np.round(tier.end_time * sampling_rate / hop_length)
                - np.round(last_end * sampling_rate / hop_length)
            )
        )
        phones.append('sp')
        end_time.append(tier.end_time)
    return phones, durations, end_time

def refine_from_labels(phones, durations, label):
    gt_phones = label.strip().split(" ")
    i = 0
    j = 0
    refined_phones = []
    gtph = None
    while i<len(phones) and j<len(gt_phones):
        ph = phones[i]
        gtph = gt_phones[j]
        if ph == gtph or gtph.lower() == ph.lower():
            i += 1
            j += 1
            refined_phones.append(gtph)
        elif ph in silence_symbol:
            i += 1
            refined_phones.append(ph)
        elif gtph in pu_symbols:
            if i > 0 and refined_phones[-1] in silence_symbol:
                refined_phones[-1] = gtph
            else:
                print("skip symbol", gtph)
            j += 1
        else:
            assert False
    if i != len(phones):
        refined_phones += phones[i:]
        print("label missing", phones[i:])
    if gtph in pu_symbols and refined_phones[-1] in silence_symbol:
        refined_phones[-1] = gtph
        j +=1
    if j != len(gt_phones):
        print("skip symbol",gt_phones[j:])


    assert len(refined_phones) == len(phones)
    return refined_phones

def remove_dup(phs, dur):
    new_phos = []
    new_gtdurs = []
    last_ph = None
    for ph, dur in zip(phs, dur):
        if ph != last_ph:
            new_phos.append(ph)
            new_gtdurs.append(dur)
        else:
            new_gtdurs[-1] += dur
        last_ph = ph
    return new_phos, new_gtdurs

def refine(phones, durations):
    phones, durations = remove_dup(phones, durations)
    for idx in range(len(phones)):
        ph = phones[idx]
        dur = durations[idx]
        if ph in silence_symbol:
            phones[idx] = get_sp(dur, idx == len(phones)-1 and phones[idx-1] not in silence_symbol, idx==0)

    return phones,durations

label_refine = False
lang = "zh"
with open(f"filelists/{lang}.dur", "w") as out_file:
    for spk in tqdm.tqdm(os.listdir(f"mfa_temp/textgrids/{lang}")):
        if os.path.isdir(f"mfa_temp/textgrids/{lang}/{spk}"):
            align_root= f"mfa_temp/textgrids/{lang}/{spk}"
            for txgridname in sorted(os.listdir(align_root)):
                if txgridname.endswith("Grid"):
                    textgrid = tgt.io.read_textgrid(f"{align_root}/{txgridname}")
                    phone, duration, end_times = get_alignment(
                        textgrid.get_tier_by_name("phones")
                    )
                    id_ = txgridname.replace(".TextGrid", "")
                    # try:
                    if label_refine:
                        label = open(f"mfa_temp/wavs/{lang}/{spk}/{id_}.txt").read()
                        phone = refine_from_labels(phone, duration, label)
                    else:
                        phone, duration = refine(phone, duration)
                    # except:
                    #     print(align_root, txgridname)
                    #     continue
                    ph = " ".join(phone)
                    du = " ".join([str(i) for i in duration])
                    ph = ph.replace("JA", ".")
                    # ph = ph.replace("JA", ".")
                    out_file.write(f"{spk}|{id_}|{ph}|{du}\n")
