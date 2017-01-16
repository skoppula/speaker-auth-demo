import numpy as np
import os
import sys

tmp_dir = sys.argv[1]
save_dir = sys.argv[2]
save_dir == save_dir if save_dir[-1] == '/' else save_dir + '/'

print("argument one: i-vector ark file:", tmp_dir)
print("argument two: saving numpy in:", save_dir)

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

num_ivecs = sum(1 for line in open(tmp_dir))
IVEC_LEN = 600
ivecs = np.zeros((num_ivecs, IVEC_LEN))
lbls = [None]*num_ivecs

with open(tmp_dir, 'r') as f:
    for i, line in enumerate(f):
        ivec_pts = line.split('[ ')
        lbls[i] = ivec_pts[0].strip().split('-')[0]
        ivecs[i] = np.fromstring(ivec_pts[1][:-2], sep=' ')
print("i-vector matrix size", np.shape(ivecs), "labels size", np.shape(lbls))

mkdir(save_dir)
np.save(save_dir + "X.npy", ivecs)
np.save(save_dir + "y.npy", lbls)
