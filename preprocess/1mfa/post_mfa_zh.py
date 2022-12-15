
import os
from preprocess.config import spk, sid
import numpy as np
import tgt


sil_phones = ["sil", "sp", "spn"]
pu_symbols = [',', '.', '!', '?', '…']
err_num = 0
all_symbols = set()
[[all_symbols.add(j) for j in i.strip().split(" ")[1:]] for i in open("mandarin_pinyin.dict").readlines()]
all_pinyin = [i.split(" ")[0] for  i in open("mandarin_pinyin.dict").readlines()]

align_root= f"preprocess/1mfa/mfa_result/{spk}"
sampling_rate = 44100
hop_length = 512
def get_sp(frames):
    # if frames <20:
    #     return "、"
    # elif frames <40:
    #     return "，"
    # elif frames <70:
    #     return "。"
    # else:
    #     return "..."
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

with open(f"preprocess/temp/{spk}.dur", "w") as out_file:
    for txgridname in sorted(os.listdir(align_root)):
        if txgridname.endswith("Grid"):
            # print(f"{align_root}/{txgridname}")
            textgrid = tgt.io.read_textgrid(f"{align_root}/{txgridname}")
            phone, duration, end_times = get_alignment(
                textgrid.get_tier_by_name("phones")
            )
            name = txgridname.replace("TextGrid", "wav")
            ph = " ".join(phone)
            du = " ".join([str(i) for i in duration])
            # for p, d in zip(ph.split(" "), du.split(" ")):
            #
            #     print(p, d)
            # break
            print(ph)
            print(du)
            wav_path = f"preprocess/1mfa/mfa_dataset/{spk}/{name}"
            out_file.write(f"{wav_path}|{ph}|{du}|{sid}\n")
            # data_count += 1
            # # print(wav_path)
            # if os.path.exists(wav_path):
            #     result_str += f"{wav_path}|{ph}|{du}|99\n"
