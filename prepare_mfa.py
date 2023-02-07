import os
import shutil

import librosa
import soundfile
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

lang = "zh"
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
    with ProcessPoolExecutor(max_workers=int(cpu_count()) // 2) as executor:
        for spk in os.listdir(f"data/{lang}"):
            if os.path.exists(f"data/{lang}/{spk}/transcription_raw.txt"):
                os.makedirs(f"mfa_temp/wavs/{lang}/{spk}", exist_ok=True)
                lines = open(f"data/{lang}/{spk}/transcription_raw.txt").readlines()
                futures = [executor.submit(process_text, line) for line in lines]
                for x in tqdm.tqdm(as_completed(futures), total=len(lines)):
                    id_, phones = x._result
                    try:
                        wav, sr = librosa.load(f"data/{lang}/{spk}/wavs/{id_}.wav", sr=44100)
                        soundfile.write(f"mfa_temp/wavs/{lang}/{spk}/{id_}.wav", wav, sr)
                        with open(f"mfa_temp/wavs/{lang}/{spk}/{id_}.txt", "w") as o:
                            o.write(phones + "\n")
                    except:
                        print("err:",spk, id_)
                    # result = f.result()
                    # o.write(result)
    #．
    # for line in open("/Volumes/Extend/下载/jsut_ver1.1 2/basic5000/transcript_utf8.txt").readlines():
    #     id_, text = line.strip().split(":")
    #     phones = text_to_phones(f"[JA]{text}[JA]")
    #     phones = " ".join(phones)
    #     with open(f"mfa_temp/wavs/jsut/{id_}.txt", "w") as o:
    #         o.write(phones + "\n")
    print("rm -rf ./mfa_temp/temp; mfa align mfa_temp/wavs/zh mfa_temp/zh_dict.dict mfa_temp/aishell3_model.zip mfa_temp/textgrids/zh --clean --overwrite -t ./mfa_temp/temp -j 5")
    print("rm -rf ./mfa_temp/temp; mfa train mfa_temp/wavs/ja/ mfa_temp/ja_dict.dict mfa_temp/model.zip mfa_temp/textgrids/ja --clean --overwrite -t ./mfa_temp/temp -j 5")
