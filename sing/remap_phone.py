from text import cleaned_text_to_sequence
sing_initials = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'w', 'x', 'y', 'z', 'zh']
sing_finals = ['E', 'En', 'a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 'i', 'i0', 'ia', 'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'ir', 'iu', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'ui', 'un', 'uo', 'v', 'van', 've', 'vn']

special_map = {
    "w": "u1",
    "y": "i1",
    "E": "ie1",
    "iu": "iou1",
    "i0": "ii1",
    "ui": "uei1",
    "un": "uen1",
    "En": "ian1",
    "SP": "sp",
    "AP": "sil",
    "a": "a2"
}

def remap_single_phone(phone):
    if phone in special_map.keys():
        return special_map[phone]
    elif phone in sing_initials:
        return phone
    elif phone in sing_finals:
        return phone+"1"
    else:
        raise Exception

def remap_phone_seq(phone_seq):
    return [remap_single_phone(phone) for phone in phone_seq]

def singing_phone_to_sequence(singing_phones):
    return cleaned_text_to_sequence(remap_phone_seq(singing_phones))

