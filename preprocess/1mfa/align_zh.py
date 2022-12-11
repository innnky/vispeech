import os
from preprocess.config import spk
MFA_PATH="preprocess/1mfa/montreal-forced-aligner/bin"
DICT_ZH = 'preprocess/dict/dict2.dict'
MODEL_DIR_ZH ="preprocess/1mfa/aishell3_model.zip"

CMD = 'mfa_align' + ' ' + str(
        f"preprocess/1mfa/mfa_dataset/{spk}") + ' ' + DICT_ZH + ' ' + MODEL_DIR_ZH + ' ' + str(f"preprocess/1mfa/mfa_result/{spk}")
os.environ['PATH'] = MFA_PATH + '/:' + os.environ['PATH']

# os.system(CMD)
print(CMD)