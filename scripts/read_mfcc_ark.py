from ark_reader import ArkReader
import numpy as np
import os

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

SCP_PATH = 'data/dd_mfcc_postvad/dd_mfcc_postvad.scp'
ar = ArkReader(SCP_PATH)

num_utts = len(ar.utt_ids)
assert num_utts == 1

utt_id, utt_data, looped = ar.read_next_utt()
NP_DIR = 'exp/'
mkdir(NP_DIR)
path = NP_DIR + utt_id + '.npy'
np.save(path, utt_data)
print("Finished saving MFCC npy", path)
