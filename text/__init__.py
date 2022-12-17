""" from https://github.com/keithito/tacotron """
import unicodedata
try:
    from paddlespeech.t2s.frontend.zh_frontend import Frontend
    frontend = Frontend()
    from g2p_en import G2p
except:
    print("failed to import text frontend")

from text.symbols import symbols
import re
from string import punctuation
pu_symbols = ['!', '?', '…', ",", "."]

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
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

def get_chinese_phonemes(text):
    res = []
    try:
        text = text.replace("嗯", "恩")
        res += frontend.get_phonemes(text)[0]
    except:
        pass
    return res



def preprocess_chinese(text):

    text = pu_symbol_replace(text)
    phonemes = []
    seg = ""
    for ch in text:
        if ch in pu_symbols:
            phonemes += get_chinese_phonemes(seg)
            seg = ""
            phonemes.append(ch)
        else:
            seg+=ch
    phonemes+=get_chinese_phonemes(seg)

    for i in range(len(phonemes)):
        if phonemes[i] not in symbols+pu_symbols:
            phonemes[i] = "sp"

    return phonemes


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon
lexicon = read_lexicon("text/en_dict.dict")

def preprocess_english(text):
    text = text.rstrip(punctuation)

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            for ch in w:
                if ch != " ":
                    phones += g2p(ch)
    phones = "}{".join(phones)
    phones = re.sub(r"\{[^\w\s]?\}", "{sil}", phones)
    phones = phones.replace("}{", " ")

    phones = phones.split(" ")

    return phones

def cleaned_text_to_sequence(cleaned_text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
  return sequence



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
def text_to_sequence(text):
    phones = text_to_phones(text)
    return cleaned_text_to_sequence(phones)



def text_to_phones(text):
    segs = mf.get_segment(text)
    phones = []
    for seg in segs:
        if seg[1] in ["zh","other"]:
            phones += preprocess_chinese(seg[0])
        elif seg[1] == "en":
            phones += preprocess_english(seg[0])
    print(phones)
    return phones
if __name__ == '__main__':
    text = "奥大家好33啊我是Ab3s,?萨达撒abst 123、~~、、 但是、、、A B C D!"
    # text = "嗯？什么东西…沉甸甸的…下午1:00，今天是2022/5/10"
    # text = "早上好，今天是2020/10/29，最低温度是-3°C。"
    # text = "…………"
    print(text_to_sequence(text))

    # print(preprocess_english("A b c d"))