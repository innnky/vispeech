import tgt
import numpy as  np

sampling_rate = 22050
hop_length = 256

sil_phones = ["sil", "sp", "spn"]
# sil_phones = []



def get_alignment(tier):

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trim leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            # For ordinary phones
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            # For silent phones
            phones.append(p)

        durations.append(
            int(
                np.round(e * sampling_rate / hop_length)
                - np.round(s * sampling_rate / hop_length)
            )
        )

    # Trim tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]

    return phones, durations, start_time, end_time


textgrid = tgt.io.read_textgrid("/Users/xingyijin/Downloads/TextGrid/LJSpeech/LJ001-0002.TextGrid")
phone, duration, start, end = get_alignment(
    textgrid.get_tier_by_name("phones")
)
print(phone, duration, start, end)

# print(textgrid)