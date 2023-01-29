# from text import symbols
# from text import cleaned_text_to_sequence
# print(cleaned_text_to_sequence(["a."]))
# import json
# import os
# spk = "engmale"
#
# trdict = {}
# # print(trs)
# for line in open("data/engmale/6097_manifest_clean_train.json").readlines():
#     tr = json.loads(line.strip())
#     audio_filepath = tr["audio_filepath"]
#     text_normalized = tr["text"]
#     id_ = audio_filepath.split("/")[-1].split(".")[0]
#     trdict[id_] = text_normalized
# with open(f'data/{spk}/transcription_raw.txt', "w") as o:
#     for wavpath in os.listdir(f"data/{spk}/wavs"):
#         id_ = wavpath.split(".")[0]
#         try:
#             text = trdict[id_]
#         except:
#             # os.system(f"rm data/{spk}/wavs/{wavpath}")
#             continue
#         o.write(f"{id_}|[EN]{text}[EN]\n")




