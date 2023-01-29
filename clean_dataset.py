# import os
#
# import tqdm
#
# from text.cleaner import text_to_phones
# for spk in os.listdir("data"):
#     if os.path.exists(f"data/{spk}/transcription_raw.txt"):
#         with open(f"data/{spk}/transcriptions.txt", "w") as o:
#             for line in tqdm.tqdm(open(f"data/{spk}/transcription_raw.txt").readlines()):
#                 id_, text = line.strip().split("|")
#                 phones = text_to_phones(text)
#                 phones = " ".join(phones)
#                 o.write(f"{id_}|啊|{phones}|rest|0|0|0\n")

import os
import tqdm
from multiprocessing import Pool
from text.cleaner import text_to_phones
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
from multiprocessing import cpu_count

def process_text(line):
    id_, text = line.strip().split("|")
    phones = text_to_phones(text)
    phones = " ".join(phones)
    return f"{id_}|啊|{phones}|rest|0|0|0\n"

if __name__ == '__main__':
    for spk in os.listdir("data"):
        if os.path.exists(f"data/{spk}/transcription_raw.txt"):
            with open(f"data/{spk}/transcriptions.txt", "w") as o:
                with ProcessPoolExecutor(max_workers=int(cpu_count()) // 2) as executor:
                    lines = open(f"data/{spk}/transcription_raw.txt").readlines()
                    futures = [executor.submit(process_text, line) for line in lines]
                    for f in tqdm.tqdm(as_completed(futures), total=len(lines)):
                        result = f.result()
                        o.write(result)

