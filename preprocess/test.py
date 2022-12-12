


singdict = {line.strip().split('\t')[0]:line.strip().split('\t')[1].split(" ") for line in open("preprocess/dict/opencpop-strict.txt").readlines()}
# enttsdict = {line.strip().split('\t')[0]:line.strip().split('\t')[1].split(" ") for line in open("preprocess/dict/en_dict.dict").readlines()}
#
#
#
ttsdict = {line.strip().split('\t')[0]:line.strip().split('\t')[1].split(" ") for line in open("preprocess/dict/dict2.dict").readlines()}

#
tts_initials = set()
tts_finals = set()
sing_initials = set()
sing_finals = set()
#
# for v in ttsdict.values():
#     tts_initials.add(v[0])
#     tts_finals.add(v[1])
#
for v in singdict.values():
    if len(v) == 1:
        sing_finals.add(v[0])
    else:
        sing_initials.add(v[0])
        sing_finals.add(v[1])
for v in ttsdict.values():
    if len(v) == 1:
        tts_finals.add(v[0])
    else:
        tts_initials.add(v[0])
        tts_finals.add(v[1])

#
print(sorted(list(tts_initials)))
print(sorted(list(sing_initials)))
print(sorted(list(tts_finals)))
print(sorted(list(sing_finals)))



#
for ph in sing_finals:
    if ph+"1" not in tts_finals:
        print(ph)
# with open("preprocess/dict/dict2.dict", "w") as o:
#     for k, v in ttsdict.items():
#         if v[0] == "":
#             v = [v[1]]
#         o.write("{}\t{}\n".format(k, " ".join(v)))
#
# silence = ["sp", "sil", "spn"]
# en_symbols = set()
# for k, v in enttsdict.items():
#     for ph in v:
#         if ph not in silence:
#             en_symbols.add(ph)
# print(en_symbols)
#
# zh_symbols = set()
# for k, v in ttsdict.items():
#     for ph in v:
#         if ph not in silence and ph != "":
#             zh_symbols.add(ph)
#
# print(zh_symbols)
# all_symbols = silence + list(zh_symbols)+ list(en_symbols)
# print(len(all_symbols))


with open("preprocess/dict/dict2.dict", "w") as d:
    for ph in tts_finals:
        d.write(f"{ph}\t{ph}\n")
    for ph in tts_initials:
        d.write(f"{ph}\t{ph}\n")
    d.write(f"sp\tsp\n")
