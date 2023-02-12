from collections import defaultdict
langs = ["zh", "ja"]
spk2utts = defaultdict(list)
for lang in langs:
    try:
        for line in open(f"filelists/{lang}_train.list").readlines():
            spk = line.split("|")[0]
            spk2utts[spk].append(line)
    except:
        pass
val_lines = []
train_lines = []
val_n_per_spk = 2
for spk, lines in spk2utts.items():
    val_lines+=lines[-val_n_per_spk:]
    train_lines+=lines[:-val_n_per_spk]

with open("filelists/train.txt", "w") as f:
    for line in train_lines:
        f.write(line)
with open("filelists/val.txt", "w") as f:
    for line in val_lines:
        f.write(line)
