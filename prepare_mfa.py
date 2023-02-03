import os
import shutil

import tqdm
from multiprocessing import Pool
from text.symbols import ja_symbols
from text.cleaner import text_to_phones
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
from multiprocessing import cpu_count

def process_text(line):
    id_, text = line.strip().split("|")
    phones = text_to_phones(text)
    phones = [ph.replace(".", "JA") if ph in ja_symbols else ph for ph in phones]
    phones = " ".join(phones)
    return (id_, phones)

if __name__ == '__main__':
    # for spk in os.listdir("data"):
    #     if os.path.exists(f"data/{spk}/transcription_raw.txt"):
    #         os.makedirs(f"mfa_temp/wavs/{spk}",exist_ok=True)
    #         with ProcessPoolExecutor(max_workers=int(cpu_count()) // 2) as executor:
    #             lines = open(f"data/{spk}/transcription_raw.txt").readlines()
    #             futures = [executor.submit(process_text, line) for line in lines]
    #             for x in tqdm.tqdm(as_completed(futures), total=len(lines)):
    #                 id_, phones = x._result
    #                 with open(f"mfa_temp/wavs/{spk}/{id_}.txt", "w") as o:
    #                     o.write(phones+"\n")

    for spk in os.listdir("data"):
        if os.path.exists(f"data/{spk}/transcription_raw.txt"):
            os.makedirs(f"mfa_temp/wavs/{spk}", exist_ok=True)
            with ProcessPoolExecutor(max_workers=int(cpu_count()) // 2) as executor:
                lines = open(f"data/{spk}/transcription_raw.txt").readlines()
                for line in tqdm.tqdm(lines):
                    id_, phones = process_text(line)
                    with open(f"mfa_temp/wavs/{spk}/{id_}.txt", "w") as o:
                        o.write(phones + "\n")
                    # shutil.move(f"data/{spk}/wavs/{id_}.wav", f"mfa_temp/wavs/{spk}/{id_}.wav")
                    # result = f.result()
                    # o.write(result)
    #．
    # for line in open("/Volumes/Extend/下载/jsut_ver1.1 2/basic5000/transcript_utf8.txt").readlines():
    #     id_, text = line.strip().split(":")
    #     phones = text_to_phones(f"[JA]{text}[JA]")
    #     phones = " ".join(phones)
    #     with open(f"mfa_temp/wavs/jsut/{id_}.txt", "w") as o:
    #         o.write(phones + "\n")
    print("mfa train mfa_temp/wavs/ mfa_temp/dict.dict mfa_temp/model.zip mfa_temp/textgrids/ --clean --overwrite -t ./mfa_temp/temp")
