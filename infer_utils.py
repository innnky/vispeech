from text.pitch_id import pitch_id
from text import text_to_sequence, _symbol_to_id
import torch
from pypinyin import  lazy_pinyin
from text.ph_map import ph_map
import librosa
import numpy as np

"""
from https://huggingface.co/spaces/Silentlin/DiffSinger
"""

def preprocess_word_level_input(inp):
    # Pypinyin can't solve polyphonic words
    text_raw = inp['text'].replace('最长', '最常').replace('长睫毛', '常睫毛') \
        .replace('那么长', '那么常').replace('多长', '多常') \
        .replace('很长', '很常')  # We hope someone could provide a better g2p module for us by opening pull requests.

    # lyric
    pinyins = lazy_pinyin(text_raw, strict=False)
    ph_per_word_lst = [ph_map[pinyin.strip()] for pinyin in pinyins if pinyin.strip() in ph_map]

    # Note
    note_per_word_lst = [x.strip() for x in inp['notes'].split('|') if x.strip() != '']
    mididur_per_word_lst = [x.strip() for x in inp['notes_duration'].split('|') if x.strip() != '']

    if len(note_per_word_lst) == len(ph_per_word_lst) == len(mididur_per_word_lst):
        print('Pass word-notes check.')
    else:
        print('The number of words does\'t match the number of notes\' windows. ',
              'You should split the note(s) for each word by | mark.')
        print(ph_per_word_lst, note_per_word_lst, mididur_per_word_lst)
        print(len(ph_per_word_lst), len(note_per_word_lst), len(mididur_per_word_lst))
        return None

    note_lst = []
    ph_lst = []
    midi_dur_lst = []
    is_slur = []
    for idx, ph_per_word in enumerate(ph_per_word_lst):
        # for phs in one word:
        # single ph like ['ai']  or multiple phs like ['n', 'i']
        ph_in_this_word = ph_per_word.split()

        # for notes in one word:
        # single note like ['D4'] or multiple notes like ['D4', 'E4'] which means a 'slur' here.
        note_in_this_word = note_per_word_lst[idx].split()
        midi_dur_in_this_word = mididur_per_word_lst[idx].split()
        # process for the model input
        # Step 1.
        #  Deal with note of 'not slur' case or the first note of 'slur' case
        #  j        ie
        #  F#4/Gb4  F#4/Gb4
        #  0        0
        for ph in ph_in_this_word:
            ph_lst.append(ph)
            note_lst.append(note_in_this_word[0])
            midi_dur_lst.append(midi_dur_in_this_word[0])
            is_slur.append(0)
        # step 2.
        #  Deal with the 2nd, 3rd... notes of 'slur' case
        #  j        ie         ie
        #  F#4/Gb4  F#4/Gb4    C#4/Db4
        #  0        0          1
        if len(note_in_this_word) > 1:  # is_slur = True, we should repeat the YUNMU to match the 2nd, 3rd... notes.
            for idx in range(1, len(note_in_this_word)):
                ph_lst.append(ph_in_this_word[-1])
                note_lst.append(note_in_this_word[idx])
                midi_dur_lst.append(midi_dur_in_this_word[idx])
                is_slur.append(1)
    ph_seq = ' '.join(ph_lst)

    if len(ph_lst) == len(note_lst) == len(midi_dur_lst):
        print(len(ph_lst), len(note_lst), len(midi_dur_lst))
        print('Pass word-notes check.')
    else:
        print('The number of words does\'t match the number of notes\' windows. ',
              'You should split the note(s) for each word by | mark.')
        return None
    return ph_seq, note_lst, midi_dur_lst, is_slur


def preprocess_phoneme_level_input(inp):
    ph_seq = inp['ph_seq']
    note_lst = inp['note_seq'].split()
    midi_dur_lst = inp['note_dur_seq'].split()
    # is_slur = inp['is_slur_seq'].split()
    is_slur = None
    
    print(len(note_lst), len(ph_seq.split()), len(midi_dur_lst))
    if len(note_lst) == len(ph_seq.split()) == len(midi_dur_lst):
        print('Pass word-notes check.')
    else:
        print('The number of words does\'t match the number of notes\' windows. ',
              'You should split the note(s) for each word by | mark.')
        return None
    return ph_seq, note_lst, midi_dur_lst, is_slur


def preprocess_input(inp):
    """

    :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
    :return:
    """

    input_type = inp["input_type"]
    # get ph seq, note lst, midi dur lst, is slur lst.
    if input_type == 'word':
        ret = preprocess_word_level_input(inp)
    elif input_type == 'phoneme':  # like transcriptions.txt in Opencpop dataset.
        ret = preprocess_phoneme_level_input(inp)
    else:
        print('Invalid input type.')
        return None

    if ret:
        ph_seq, note_lst, midi_dur_lst, is_slur = ret
    else:
        print('==========> Preprocess_word_level or phone_level input wrong.')
        return None

    # convert note lst to midi id; convert note dur lst to midi duration
    try:
        midis = [librosa.note_to_midi(x.split("/")[0]) if x != 'rest' else 0
                  for x in note_lst]
        midi_dur_lst = [float(x) for x in midi_dur_lst]
    except Exception as e:
        print(e)
        print('Invalid Input Type.')
        return None
    # print(ph_seq)
    ph_token = [_symbol_to_id[i] for i in ph_seq.split(" ")]
    item = { 'ph': ph_seq,
            'ph_token': ph_token, 'pitch_midi': np.asarray(midis), 'midi_dur': np.asarray(midi_dur_lst),
            'is_slur': np.asarray(is_slur), }
    item['ph_len'] = len(item['ph_token'])
    return item

# inp = {
#         'text': '小酒窝长睫毛AP是你最美的记号',
#         'notes': 'C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4',
#         'notes_duration': '0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340',
#         'input_type': 'word'
# }  # user input: Chinese characters
# res = preprocess_input(inp)
# print(res)