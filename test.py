# import tgt
# import numpy as  np
#
# sampling_rate = 22050
# hop_length = 256
#
# # sil_phones = ["sil", "sp", "spn"]
# # sil_phones = []
#
# def get_sp(frames):
#     if frames <20:
#         return "、"
#     elif frames <40:
#         return "，"
#     elif frames <70:
#         return "。"
#     else:
#         return "..."
#
#
#
# def get_alignment(tier):
#
#     phones = []
#     durations = []
#     start_time = 0
#     end_time = 0
#     last_end = 0
#     for t in tier._objects:
#         start, end, phone = t.start_time, t.end_time, t.text
#         # print(f"音素：{phone}，开始:{start}，结束:{end}")
#         # Trim leading silences
#         if last_end != start:
#             durations.append(
#                 int(
#                     np.round(start * sampling_rate / hop_length)
#                     - np.round(last_end * sampling_rate / hop_length)
#                 )
#             )
#             phones.append(get_sp(durations[-1]))
#
#         phones.append(phone)
#         durations.append(
#             int(
#                 np.round(end * sampling_rate / hop_length)
#                 - np.round(start * sampling_rate / hop_length)
#             )
#         )
#         last_end = end
#
#     if tier.end_time != last_end:
#         durations.append(
#             int(
#                 np.round(tier.end_time * sampling_rate / hop_length)
#                 - np.round(last_end * sampling_rate / hop_length)
#             )
#         )
#         phones.append(get_sp(durations[-1]))
#
#
#     return phones, durations, start_time, end_time
#
# filelist = "filelists/biaobei.txt"
#
# textgrid_root = "/Volumes/Extend/AI/MFA/aligned/biaobei"
# all_symbols = set()
# with open(filelist+".preprocessed", "w") as out:
#     with open(filelist) as f:
#         f0paths = []
#         for line in f.readlines():
#             wavpath = line.split("|")[0]
#             f0 = np.load(wavpath+".f0.npy")
#             name = line.split("|")[0].split("/")[-1].split(".")[0]
#             textgrid = tgt.io.read_textgrid(f"{textgrid_root}/{name}.TextGrid")
#             phone, duration, start, end = get_alignment(
#                 textgrid.get_tier_by_name("phones")
#             )
#             print(sum(duration), f0.shape)
#             for i in phone:
#                 all_symbols.add(i)
#             # print(phone, duration, start, end, sum(duration),textgrid.get_tier_by_name("phones").end_time * sampling_rate / hop_length)
#             duration = " ".join([str(i) for i in duration])
#             phone = " ".join(phone)
#             out.write(f"{wavpath}|{phone}|{duration}\n")
# print(all_symbols)

x = [17, 34, 71, 34, 18, 76, 72,  9, 71, 34, 18, 72, 51, 71, 66, 72, 82, 76,
         18, 76, 72, 82, 76, 18,  7, 82, 22, 17,]
print(sum(x))