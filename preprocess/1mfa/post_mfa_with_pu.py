import os

import tgt
import numpy as  np
from preprocess.config import spk, sid

sampling_rate = 44100
hop_length = 512
sil_phones = ["sil", "sp", "spn"]
pu_symbols = [',', '.', '!', '?', '…']
err_num = 0
all_symbols = set()
[[all_symbols.add(j) for j in i.strip().split(" ")[1:]] for i in open("preprocess/dict/dict2.dict").readlines()]
all_pinyin = [i.split(" ")[0] for i in open("preprocess/dict/dict2.dict").readlines()]

# sil_phones = []

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


def get_phone_starttimes(phone, duration):
    res_map = {}
    start = 0
    for p, d in zip(phone, duration):
        if p not in pu_symbols and p not in sil_phones:
            res_map[p] = start
        start += d
    return res_map

def is_chinese(word):
    return word in all_pinyin

def find_last(l, element):
    return len(l)-[i for i in reversed(l)].index(element)-1



def get_phone_dur(tg, txx):

    textgrid = tgt.io.read_textgrid(tg)
    phone, duration, end_times = get_alignment(
        textgrid.get_tier_by_name("phones")
    )

    texts = open(txx).readline().strip().split(" ")
    words = [(t.text, t.start_time) for t in textgrid.get_tier_by_name("words").intervals]
    words.append(("#end", textgrid.get_tier_by_name("words").end_time))

    before_add_duration = duration.copy()
    before_add_phone = phone.copy()

    w_i = 0
    for tx in texts:
        word, st = words[w_i]
        if tx == word:
            w_i+=1
            if tx in pu_symbols:
                assert st !=0
                idx = find_last(end_times, st)
                phone[idx+1] = tx
            continue
        if tx not in pu_symbols:
            # assert not is_chinese(tx), texts
            # 出现未知符号（既不是汉字也不是pu_symbols）
            continue
        if st == 0:
            continue
        idx = find_last(end_times, st)
        c = 0
        while idx-c >= 0 and phone[idx-c] in sil_phones:
            c += 1
        c -= 1
        if c == -1:
            if phone[idx] not in pu_symbols:
                # print()
                next_id = idx+1
                phone.insert(next_id, tx)
                duration.insert(next_id, 0)
                end_times.insert(next_id, end_times[idx])

            else:
                phone.insert(idx+1, tx)
                duration.insert(idx+1, duration[idx]//2)
                duration[idx] = duration[idx] - duration[idx]//2
                last_end_time = 0 if idx == 0 else end_times[idx-1]
                end_times.insert(idx+1,end_times[idx])
                end_times[idx] = (end_times[idx]+last_end_time)/2

        else:
            assert phone[idx-c] in sil_phones
            phone[idx-c] = tx
    assert abs(sum(duration) - textgrid.get_tier_by_name("phones").end_time * sampling_rate / hop_length)<2
    assert sum(before_add_duration) == sum(duration)
    assert get_phone_starttimes(before_add_phone, before_add_duration) == get_phone_starttimes(phone, duration)
    return phone, duration


#将拼音声调分割开来，减少symbol数量
def sep_tone(phones, durations):
    new_phones = []
    new_durations = []
    for ph,dur in zip(phones,durations):
        if ph[-1] in ["1", "2", "3", "4", "5"]:
            new_phones.append(ph[:-1])
            new_durations.append(dur)
            new_phones.append(ph[-1])
            new_durations.append(0)
        else:
            new_phones.append(ph)
            new_durations.append(dur)
    return new_phones, new_durations

def preprocess_speaker(outname, align_root, wav_root):
    with open(outname, "w") as out:
        for tg in sorted(os.listdir(align_root)):
            name = tg.split(".")[0]
            if len(name)>1:
                # try:
                phone, duration = get_phone_dur(f"{align_root}/{name}.TextGrid",  f"{wav_root}/{name}.lab")
                phone, duration = sep_tone(phone, duration )

                phone = " ".join(phone)
                duration = " ".join([str(i) for i in duration])
                print(tg,phone, duration)
                out.write(f"{wav_root}/{name}.wav|{phone}|{duration}|{sid}\n")
                # except:
                #     print(open(f"{wav_root}/{name}.txt").readline().strip())


outname = f"preprocess/temp/{spk}.dur"
align_root = f"preprocess/1mfa/mfa_result/{spk}"
wav_root = f"preprocess/1mfa/mfa_dataset/{spk}"
preprocess_speaker(outname, align_root, wav_root)

# #
# outname = "paimon.txt.preprocessd"
# align_root = "/Volumes/Extend/AI/MFA/aligned/VO_paimon/VO_paimon"
# wav_root = "/Volumes/Extend/AI/数据集/VO_paimon"
# preprocess_speaker(outname, align_root, wav_root)
# #
# outname = "mxj.txt.preprocessd"
# align_root = "/Volumes/Extend/AI/MFA/aligned/mxj/mxj"
# wav_root = "/Volumes/Extend/AI/MFA/dataset/mxj"
# preprocess_speaker(outname, align_root, wav_root)

# textgrid = tgt.io.read_textgrid("/Volumes/Extend/AI/MFA/aligned/VO_paimon/VO_paimon/vo_CYAQ001_1_paimon_01.TextGrid")
# phone, duration, end_times = get_alignment(
#     textgrid.get_tier_by_name("phones")
# )


# print(phone)
