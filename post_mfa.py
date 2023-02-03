
import os
import numpy as np
import tgt



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


with open(f"filelists/files.dur", "w") as out_file:
    for spk in os.listdir("mfa_temp/textgrids"):
        if os.path.isdir(f"mfa_temp/textgrids/{spk}"):
            align_root= f"mfa_temp/textgrids/{spk}"
            for txgridname in sorted(os.listdir(align_root)):
                if txgridname.endswith("Grid"):
                    # print(f"{align_root}/{txgridname}")
                    textgrid = tgt.io.read_textgrid(f"{align_root}/{txgridname}")
                    phone, duration, end_times = get_alignment(
                        textgrid.get_tier_by_name("phones")
                    )
                    id_ = txgridname.replace(".TextGrid", "")
                    ph = " ".join(phone)
                    du = " ".join([str(i) for i in duration])
                    out_file.write(f"{spk}|{id_}|{ph}|{du}\n")
