
import os
import numpy as np
import tgt
from text.symbols import pu_symbols

silence_symbol = ["sil", "sp", "spn"]
sampling_rate = 44100
hop_length = 512


def get_sp(frames):
    return "sp"

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
            phones.append(get_sp(durations[-1]))
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
        phones.append(get_sp(durations[-1]))
        end_time.append(tier.end_time)
    return phones, durations, end_time

def refine_and_add_symbol(phones, durations, label):
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
with open(f"filelists/files.dur", "w") as out_file:
    for spk in os.listdir("mfa_temp/textgrids"):
        if os.path.isdir(f"mfa_temp/textgrids/{spk}"):
            align_root= f"mfa_temp/textgrids/{spk}"
            for txgridname in sorted(os.listdir(align_root)):
                if txgridname.endswith("Grid"):
                    textgrid = tgt.io.read_textgrid(f"{align_root}/{txgridname}")
                    phone, duration, end_times = get_alignment(
                        textgrid.get_tier_by_name("phones")
                    )
                    id_ = txgridname.replace(".TextGrid", "")
                    label = open(f"mfa_temp/wavs/{spk}/{id_}.txt").read()
                    try:
                        phone = refine_and_add_symbol(phone, duration, label)
                    except:
                        print(align_root, txgridname)
                    ph = " ".join(phone)
                    du = " ".join([str(i) for i in duration])
                    ph = ph.replace("JA", ".")
                    # ph = ph.replace("JA", ".")
                    out_file.write(f"{spk}|{id_}|{ph}|{du}\n")
