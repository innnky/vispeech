# modified from https://github.com/CjangCjengh/vits/blob/main/text/japanese.py
import re
from unidecode import unidecode
import pyopenjtalk


from text.symbols import remove_invalid_phonemes,pu_symbols, symbols

# Regular expression matching Japanese without punctuation marks:
_japanese_characters = re.compile(
    r'[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]')

# Regular expression matching non-Japanese characters or punctuation marks:
_japanese_marks = re.compile(
    r'[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]')

# List of (symbol, Japanese) pairs for marks:
_symbols_to_japanese = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('％', 'パーセント')
]]

# List of (romaji, ipa) pairs for marks:
_romaji_to_ipa = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('ts', 'ʦ'),
    ('u', 'ɯ'),
    ('j', 'ʥ'),
    ('y', 'j'),
    ('ni', 'n^i'),
    ('nj', 'n^'),
    ('hi', 'çi'),
    ('hj', 'ç'),
    ('f', 'ɸ'),
    ('I', 'i*'),
    ('U', 'ɯ*'),
    ('r', 'ɾ')
]]

# List of (romaji, ipa2) pairs for marks:
_romaji_to_ipa2 = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('u', 'ɯ'),
    ('ʧ', 'tʃ'),
    ('j', 'dʑ'),
    ('y', 'j'),
    ('ni', 'n^i'),
    ('nj', 'n^'),
    ('hi', 'çi'),
    ('hj', 'ç'),
    ('f', 'ɸ'),
    ('I', 'i*'),
    ('U', 'ɯ*'),
    ('r', 'ɾ')
]]

# List of (consonant, sokuon) pairs:
_real_sokuon = [(re.compile('%s' % x[0]), x[1]) for x in [
    (r'Q([↑↓]*[kg])', r'k#\1'),
    (r'Q([↑↓]*[tdjʧ])', r't#\1'),
    (r'Q([↑↓]*[sʃ])', r's\1'),
    (r'Q([↑↓]*[pb])', r'p#\1')
]]

# List of (consonant, hatsuon) pairs:
_real_hatsuon = [(re.compile('%s' % x[0]), x[1]) for x in [
    (r'N([↑↓]*[pbm])', r'm\1'),
    (r'N([↑↓]*[ʧʥj])', r'n^\1'),
    (r'N([↑↓]*[tdn])', r'n\1'),
    (r'N([↑↓]*[kg])', r'ŋ\1')
]]


def symbols_to_japanese(text):
    for regex, replacement in _symbols_to_japanese:
        text = re.sub(regex, replacement, text)
    return text


def preprocess_jap(text):
    '''Reference https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html'''
    text = symbols_to_japanese(text)
    sentences = re.split(_japanese_marks, text)
    marks = re.findall(_japanese_marks, text)
    text = []
    for i, sentence in enumerate(sentences):
        if re.match(_japanese_characters, sentence):
            p = pyopenjtalk.g2p(sentence)
            text += p.split(" ")

        if i < len(marks):
            text += [marks[i].replace(' ', '')]
    return text


def ja_to_phonemes(text):
    jap_phs = preprocess_jap(text)
    jap_phs = [i+"JA" if i not in pu_symbols+["pau"] else i for i in jap_phs if i !=""]
    for i in jap_phs:
        if i not in symbols:
            print(jap_phs)
            print("debug jap: missing ", i)
    return remove_invalid_phonemes(jap_phs)