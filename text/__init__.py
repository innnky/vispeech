import time
t = time.time()
from text.symbols import symbols

from text.mix_frontend import mf, preprocess_chinese, preprocess_english

_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def cleaned_text_to_sequence(cleaned_text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
  return sequence



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
    text = "借还款,他只是一个纸老虎，开户行，奥大家好33啊我是Ab3s,?萨达撒abst 123、~~、、 但是、、、A B C D!"
    # text = "嗯？什么东西…沉甸甸的…下午1:00，今天是2022/5/10"
    # text = "早上好，今天是2020/10/29，最低温度是-3°C。"
    # text = "…………"
    print(text_to_sequence(text))

    print(time.time()-t)