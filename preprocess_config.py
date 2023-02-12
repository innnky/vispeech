import json

config = json.load(open("configs/config.json"))
spk2id = {}
sid = 0
for line in open(f"filelists/train.list").readlines():
    spk = line.split("|")[0]
    if spk not in spk2id.keys():
        spk2id[spk] = sid
        sid+=1

config["data"]['spk2id'] = spk2id

with open("configs/config.json", "w")as f:
    json.dump(config,f, indent=2)
