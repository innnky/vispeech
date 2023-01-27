import re

from text.en_frontend import en_to_phonemes
from text.ja_frontend import ja_to_phonemes
from text.zh_frontend import zh_to_phonemes

_japanese_characters = re.compile(
    r'[\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]')

def others_to_phonemes(text):
    # print(text)
    if text == '':
        return []
    segs = mf.get_segment(text)
    phones = []
    for seg in segs:
        if seg[1] in ["zh","other"]:
            phones += zh_to_phonemes(seg[0])
        elif seg[1] == "en":
            phones += en_to_phonemes(seg[0])
        elif seg[1] == "ja":
            phones += ja_to_phonemes(seg[0])
    return phones

from string import punctuation
pu_symbols = ['!', '?', '…', ",", "."]

# print(_symbol_to_id)

def str_replace( data):
    chinaTab = ['：', '；', '，', '。', '！', '？', '【', '】', '“', '（', '）', '%', '#', '@', '&', "‘", ' ', '\n', '”',"—", "·",'、']
    englishTab = [':', ';', ',', '.', '!', '?', '[', ']', '"', '(', ')', '%', '#', '@', '&', "'", ' ', '', '"', "-", "-", ","]
    for index in range(len(chinaTab)):
        if chinaTab[index] in data:
            data = data.replace(chinaTab[index], englishTab[index])
    return data

def pu_symbol_replace(data):
    chinaTab = ['！', '？', "…", "，", "。"]
    englishTab = ['!', '?', "…", ",", "."]
    for index in range(len(chinaTab)):
        if chinaTab[index] in data:
            data = data.replace(chinaTab[index], englishTab[index])
    return data



# from paddle speech https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/paddlespeech/t2s/frontend/mix_frontend.py
class MixFrontend:
    def __init__(self):
        pass
    def is_chinese(self, char):
        if '\u4e00' <= char <= '\u9fa5' or "0"<=char<="9" or char in punctuation:
            return True
        else:
            return False

    def is_alphabet(self, char):
        if ('\u0041' <= char <= '\u005a') or ('\u0061' <= char <= '\u007a'):
            return True
        else:
            return False

    def is_japanese(self, char):
        return re.match(_japanese_characters, char)

    def is_other(self, char):
        if not (self.is_chinese(char) or self.is_alphabet(char)):
            return True
        else:
            return False

    def get_segment(self, text: str):
        # sentence --> [ch_part, en_part, ch_part, ...]
        text = str_replace(text)
        segments = []
        types = []
        flag = 0
        temp_seg = ""
        temp_lang = ""

        # Determine the type of each character. type: blank, chinese, alphabet, number, unk and point.
        for ch in text:
            if self.is_chinese(ch):
                types.append("zh")
            elif self.is_alphabet(ch):
                types.append("en")
            elif self.is_japanese(ch):
                types.append("ja")
            else:
                types.append("other")

        assert len(types) == len(text)

        for i in range(len(types)):
            # find the first char of the seg
            if flag == 0:
                temp_seg += text[i]
                temp_lang = types[i]
                flag = 1

            else:
                if temp_lang == "other":
                    if types[i] == temp_lang:
                        temp_seg += text[i]
                    else:
                        temp_seg += text[i]
                        temp_lang = types[i]

                else:
                    if types[i] == temp_lang:
                        temp_seg += text[i]
                    elif types[i] == "other":
                        temp_seg += text[i]
                    else:
                        segments.append((temp_seg, temp_lang))
                        temp_seg = text[i]
                        temp_lang = types[i]
                        flag = 1

        segments.append((temp_seg, temp_lang))

        return segments

mf = MixFrontend()

