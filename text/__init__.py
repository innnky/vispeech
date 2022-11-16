""" from https://github.com/keithito/tacotron """
from text.symbols import symbols
import jieba
import cn2an
import re
from pypinyin import pinyin, lazy_pinyin, Style

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

pu_symbols = [',', '.', '!', '?', '…', '-', '~']
all_pinyin = [i.split("\t")[0] for i in open("text/dict.dict").readlines() if i.split("\t")[0] not in pu_symbols]
pinyin2ph = {i.split("\t")[0]:i.split("\t")[-1].strip().split(" ") for  i in open("text/dict.dict").readlines()}

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
    chinaTab = ['：', '；', '，', '。', '！', '？', '【', '】', '“', '（', '）', '%', '#', '@', '&', "‘", ' ', '\n', '”',"—", "·"]
    englishTab = [':', ';', ',', '.', '!', '?', '[', ']', '"', '(', ')', '%', '#', '@', '&', "'", ' ', '', '"', "-","-"]
    for index in range(len(chinaTab)):
        if chinaTab[index] in data:
            data = data.replace(chinaTab[index], englishTab[index])
    return data

def clean(text):
    rt = text
    text = number_to_chinese(text)
    text = chinese_to_bopomofo(text)
    # text = unicodedata.normalize('NFKD', text)
    text = str_replace(text)
    debug = [ch for ch in text.split(" ") if ch not in all_pinyin and ch not in pu_symbols]
    if len(debug)>1:
        print(debug, rt)
    ch = " ".join([ch for  ch in text.split(" ") if ch in all_pinyin or ch in pu_symbols])
    return ch.strip()

def text_to_sequence(text):
  pinyins = clean(text).split(" ")
  phones = []
  for pyin in pinyins:
      phones += pinyin2ph[pyin]
  return cleaned_text_to_sequence(phones)


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

