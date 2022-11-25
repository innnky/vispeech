""" from https://github.com/keithito/tacotron """
import unicodedata

from text.symbols import symbols
import jieba
import cn2an
import re
from pypinyin import pinyin, lazy_pinyin, Style
from g2p_en import G2p
from string import punctuation
# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

pu_symbols = [',', '.', '!', '?', '…', '-', '~']
pre = "text/"
all_pinyin = [i.split("\t")[0] for i in open(pre+"zh_dict.dict").readlines() if i.split("\t")[0] not in pu_symbols]
pinyin2ph = {i.split("\t")[0]:i.split("\t")[-1].strip().split(" ") for  i in open(pre+"zh_dict.dict").readlines()}

def number_to_chinese(text):
    numbers = re.findall(r'\d+(?:\.?\d+)?', text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    return text

def chinese_to_bopomofo(text):
    text = text.replace('、', '，').replace('；', '，').replace('：', '，')
    words = jieba.lcut(text, cut_all=False)
    text = ''
    for word in words:
        pinyins = lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True,strict=False)
        for p in pinyins:
            if p in ["n1", "n2", "n3", "n4"]:
                p = p.replace("n", "en")
            text += ' '+p
    return text.strip()

def str_replace( data):
    chinaTab = ['：', '；', '，', '。', '！', '？', '【', '】', '“', '（', '）', '%', '#', '@', '&', "‘", ' ', '\n', '”',"—", "·",'、']
    englishTab = [':', ';', ',', '.', '!', '?', '[', ']', '"', '(', ')', '%', '#', '@', '&', "'", ' ', '', '"', "-", "-", ","]
    for index in range(len(chinaTab)):
        if chinaTab[index] in data:
            data = data.replace(chinaTab[index], englishTab[index])
    return data

def clean_zh(text):
    rt = text
    text = number_to_chinese(text)
    text = chinese_to_bopomofo(text)
    # text = unicodedata.normalize('NFKD', text)
    text = str_replace(text)
    ch = " ".join([ch for  ch in text.split(" ") if ch in all_pinyin or ch in pu_symbols])
    return ch.strip()

def preprocess_chinese(text):
  pinyins = clean_zh(text).split(" ")
  phones = []
  for pyin in pinyins:
      if pyin in pu_symbols:
          phones.append(pyin)
      else:
          try:
            phones += pinyin2ph[pyin]
          except:
            pass
  return phones

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
lexicon = read_lexicon(pre+"en_dict.dict")

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


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


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
    segs = mf.get_segment(text)
    phones = []
    for seg in segs:
        if seg[1] == "zh":
            phones += preprocess_chinese(seg[0])
        elif seg[1] == "en":
            phones += preprocess_english(seg[0])
    # print(phones)
    return cleaned_text_to_sequence(phones)
if __name__ == '__main__':
    text = "大家好33啊我是Ab3s,?萨达撒abst 123、、、 但是、、、A B C D"
    print(text_to_sequence(text))

    # print(preprocess_english("A b c d"))