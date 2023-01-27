
from text.frontend.zh_frontend import Frontend
from text.symbols import remove_invalid_phonemes
frontend = Frontend()


pu_symbols = ['!', '?', '…', ",", "."]

# print(_symbol_to_id)


def pu_symbol_replace(data):
    chinaTab = ['！', '？', "…", "，", "。",'、', "..."]
    englishTab = ['!', '?', "…", ",", ".",",", "…"]
    for index in range(len(chinaTab)):
        if chinaTab[index] in data:
            data = data.replace(chinaTab[index], englishTab[index])
    return data

# def del_special_pu(data):
#     ret = ''
#     to_del = ["'", "\"", "“","", '‘', "’", "”"]
#     for i in data:
#         if i not in to_del:
#             ret+=i
#     return ret


def zh_to_phonemes(text):
    # 替换标点为英文标点
    text = pu_symbol_replace(text)
    phones = frontend.get_phonemes(text)[0]
    return remove_invalid_phonemes(phones)
