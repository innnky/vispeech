import os.path
pu_symbols = ['!', '?', '…', ",", ".", '-']
ja_symbols = [
    # japanese-common
    'ts.', 'f.', 'sh.', 'ry.', 'py.', 'h.', 'p.', 'N.', 'a.', 'm.', 'w.', 'ky.',
    'n.', 'd.', 'j.', 'cl.', 'ny.', 'z.', 'o.', 'y.', 't.', 'u.', 'r.', 'pau',
    'ch.', 'e.', 'b.', 'k.', 'g.', 's.', 'i.',
    # japanese-unique
    'gy.', 'my.', 'hy.', 'br', 'by.', 'v.', 'ty.', 'xx.', 'U.', 'I.', 'dy.'
]

op_symbols = [
    'a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 'iii', 'ii', 'i', 'ia', 'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'iou', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'uei', 'uen', 'ueng', 'uo', 'v', 'van', 've', 'vn', 'AH', 'AA', 'AO', 'ER', 'IH', 'IY', 'UH', 'UW', 'EH', 'AE', 'AY', 'EY', 'OY', 'AW', 'OW', 'AH3', 'AA3', 'AO3', 'ER3', 'IH3', 'IY3', 'UH3', 'UW3', 'EH3', 'AE3', 'AY3', 'EY3', 'OY3', 'AW3', 'OW3', 'D-1', 'T-1', 'P*', 'B*', 'T*', 'D*', 'K*', 'G*', 'M*', 'N*', 'NG*', 'L*', 'S*', 'Z*', 'Y*', 'TH*', 'DH*', 'SH*', 'ZH*', 'CH*', 'JH*', 'V*', 'W*', 'F*', 'R*', 'HH*', 'or', 'ar', 'aor', 'our', 'angr', 'eir', 'engr', 'air', 'ianr', 'iaor', 'ir', 'ingr', 'ur', 'iiir', 'uar', 'uangr', 'uenr', 'iir', 'ongr', 'uor', 'ueir', 'iar', 'iangr', 'inr', 'iour', 'vr', 'uanr', 'ruai', 'TR', 'rest', 'w', 'SP', 'AP', 'un', 'y', 'ui', 'iu', 'iour', 'vr', 'uanr', 'ruai', 'TR', 'rest', 'w', 'SP', 'AP', 'un', 'y', 'ui', 'iu', 'i0', 'E', 'En',
]

symbols =['sp', 'sil', 'spn', 'iang2', 'eir3', 'g', 'vanr4', 'eng4', 'ueir1', 'ian3', 'z', 'uor2', 'van4', 'aor4', 'or3', 'i1', 'uangr4', 'ingr4', 'ur4', 'ior1', 'iongr1', 'iangr3', 'uengr5', 'eng3', 'd', 'uengr2', 'ingr5', 'iang3', 'ang4', 'air2', 'c', 'ian5', 'ianr3', 'an4', 'iangr2', 'uai4', 'uanr4', 'iiir1', 'u4', 'van3', 'iaor1', 'iour1', 'our3', 'iangr4', 'engr2', 'air5', 'vn1', 'uar5', 'ou1', 'ai5', 'inr3', 'ar5', 'engr5', 'uor3', 'o2', 'iou4', 'uenr2', 'iongr5', 'iaor4', 'iao1', 'uei1', 'ongr3', 'h', 'e2', 'iou3', 'ua3', 'ianr5', 'u1', 'ang2', 'uo2', 'ia5', 've3', 'iar4', 'vn5', 'enr5', 'ir5', 'i3', 'iaor3', 'our2', 'ver1', 'ueir2', 'ao3', 'uan4', 'ua2', 'io5', 'io4', 'inr1', 'enr2', 'ei5', 'ii3', 'an3', 'uor4', 'iou2', 'u3', 'uangr3', 'ur3', 'o5', 'uai1', 've2', 'vr4', 'iou1', 'in1', 'ian4', 'e1', 'eng2', 'iao4', 'or1', 'ing3', 'uo3', 'vnr2', 'inr4', 'angr1', 'uan2', 'ai4', 'eng5', 'uang2', 'vanr3', 'anr1', 'ie3', 'iar1', 'uo5', 'q', 'iiir2', 'uen1', 'uen5', 'ai1', 'en2', 'uai3', 'our5', 'ingr3', 'ii5', 'uor1', 'i4', 'uar2', 'uan3', 'uengr1', 'iour4', 'aor3', 'er2', 'ianr1', 'eir2', 'van1', 'uo1', 'ueng4', 'r', 'ar2', 'iou5', 'er3', 'ar1', 'ie5', 'uangr1', 'uenr5', 'eir4', 'or4', 'ier3', 'ier1', 'iangr5', 'ch', 'air1', 'vnr1', 'uang1', 'ie1', 'ueng3', 'iir1', 'ao5', 'in4', 'van5', 'er5', 'iir5', 'anr5', 'iong1', 'ang5', 'iiir5', 'eir5', 'iar5', 'ior3', 'vr2', 'an1', 'uai2', 'zh', 'e4', 'm', 'vn4', 'an2', 'ueir4', 'uang5', 'iar3', 'ing2', 'iong5', 'ver3', 'iong3', 'ueng5', 'ei3', 'vn2', 'our4', 'x', 'ir2', 'vnr3', 'sh', 'uor5', 'iang1', 'uanr2', 'angr4', 'iii2', 'uei4', 'v1', 've4', 'uenr3', 've5', 'engr4', 'en4', 'uang4', 'ver2', 'ia2', 'ver4', 'ir1', 'ua4', 'vnr5', 'uo4', 'en1', 'or2', 'iang4', 'uan1', 'e3', 'ao4', 'ou3', 'ao2', 'our1', 'io3', 'or5', 'uenr4', 'ing4', 'ia3', 'ueng2', 'ia1', 'enr1', 'engr1', 'ar4', 'ier2', 've1', 'aor2', 'uengr4', 'ur5', 'ang1', 'ei2', 'ua1', 'ueir3', 'ian2', 'iour5', 'a3', 'i5', 'ian1', 'en5', 'uar4', 'ir3', 'l', 'uar1', 'iour3', 'v5', 'ur1', 'o3', 'ong3', 'n', 'io1', 'uangr2', 'ianr4', 'ie2', 'ii1', 'anr3', 'j', 'uanr3', 'in2', 'ou5', 'ior2', 'ingr1', 'in5', 'iiir4', 'b', 'uei2', 'iii3', 'ueng1', 'ei4', 'o4', 'inr2', 'vr3', 'ei1', 'uair5', 'ianr2', 'uen2', 'uei3', 'vr5', 'inr5', 'o1', 'eng1', 'ai3', 'iongr4', 'angr3', 'ong1', 'iangr1', 'iang5', 'aor1', 'ing5', 'iongr2', 'iiir3', 'ii4', 'ong2', 'ou2', 'en3', 'an5', 'eir1', 'v2', 'iong4', 'iir2', 'p', 'uanr5', 'uar3', 'uangr5', 't', 'ongr1', 'uair3', 'iaor2', 'io2', 'vanr1', 'e5', 'uen4', 'uan5', 'vanr2', 'u2', 'ang3', 'iii1', 'v4', 'anr4', 'engr3', 'ver5', 'ii2', 'iir3', 'uair1', 'ier4', 'er4', 'iao2', 'ou4', 'a5', 'a4', 'ongr2', 'ongr5', 'vn3', 'v3', 'iii5', 'angr5', 'i2', 'ir4', 'uengr3', 'er1', 'ior5', 'vanr5', 'ong5', 'iong2', 'ier5', 'ior4', 'air3', 'ai2', 'u5', 'iour2', 'k', 'van2', 'ar3', 'vr1', 'ia4', 'f', 'iaor5', 'ur2', 'a2', 'enr4', 'uanr1', 'iir4', 'uair2', 's', 'ua5', 'ueir5', 'a1', 'iao5', 'iar2', 'uenr1', 'vnr4', 'angr2', 'ing1', 'iao3', 'uang3', 'air4', 'ongr4', 'iongr3', 'iii4', 'in3', 'uen3', 'uei5', 'ie4', 'enr3', 'aor5', 'uai5', 'ao1', 'ong4', 'anr2', 'ingr2', 'uair4', 'N', 'AA2', 'EH0', 'AO0', 'EY1', 'AE2', 'AY0', 'ER0', 'HH', 'K', 'SH', 'AY1', 'OY2', 'M', 'UW1', 'AH2', 'AO1', 'AW0', 'AY2', 'EY2', 'F', 'JH', 'OW1', 'W', 'ER1', 'AA0', 'DH', 'OW2', 'UW2', 'OW0', 'B', 'AH0', 'IH0', 'EH1', 'AW2', 'UH0', 'ZH', 'P', 'R', 'V', 'Y', 'ER2', 'CH', 'UW0', 'UH2', 'AO2', 'IH2', 'AE1', 'NG', 'AH1', 'AE0', 'S', 'L', 'IY0', 'G', 'IY1', 'IH1', 'IY2', 'TH', 'Z', 'AW1', 'AA1', 'T', 'OY0', 'D', 'EY0', 'UH1', 'EH2', 'OY1']
symbols = symbols + pu_symbols + ja_symbols + op_symbols


def remove_invalid_phonemes(phonemes):
    # 移除未识别的音素符号
    new_phones = []
    for ph in phonemes:
        if ph in symbols:
            new_phones.append(ph)
        else:
            print("skip：", ph)
    return new_phones