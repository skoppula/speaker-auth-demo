import numpy as np
import sys

MFCC_SIZE = 20
N_INP_FRMS = 50

assert sys.argv[1]
utt_data = np.load(sys.argv[1])
n_frms = utt_data.shape[0] - N_INP_FRMS + 1
batch_x = np.zeros((n_frms, N_INP_FRMS*MFCC_SIZE))
for i in range(n_frms):
    batch_x[i] = utt_data[i:(i+N_INP_FRMS), 0:MFCC_SIZE].reshape(1, N_INP_FRMS*MFCC_SIZE)
    
def bn(inp, var_ema, mean_ema, gamma, beta):
    scale = gamma/np.sqrt(var_ema+1e-5)
    scaled_inp = inp*scale
    scaled_mean_ema = scale*mean_ema
    bias = beta-scaled_mean_ema
    out = scaled_inp + bias
    return out

def maxout(W, b, inp):
    t1 = np.dot(inp, W[0]) + b[0]
    t2 = np.dot(inp, W[1]) + b[1]
    t3 = np.dot(inp, W[2]) + b[2]
    t4 = np.dot(inp, W[3]) + b[3]
    tmax = np.maximum(np.maximum(np.maximum(t1,t2), t3), t4)
    return tmax

def fc(inp,W,b):
    return np.dot(inp,W) + b

vrs = np.load('model_data/model-1.npy').item()

inp = batch_x
for i in range(1,6):
    W = vrs['maxout_dense_%s/W' % (i,)]
    b = vrs['maxout_dense_%s/b' % (i,)]

    mx = maxout(W, b, inp)

    var_ema = vrs['batch_normalization_%s/moving_variance' % (i,)]
    mean_ema = vrs['batch_normalization_%s/moving_mean' % (i,)]
    gamma = vrs['batch_normalization_%s/gamma' % (i,)]
    beta = vrs['batch_normalization_%s/beta' % (i,)]

    inp = bn(mx, var_ema, mean_ema, gamma, beta)

W = vrs['dense_1/kernel']
b = vrs['dense_1/bias']
out = fc(inp, W, b)


g_mean = np.average(out, axis=0)
sum_dists = 0
for i in range(10):
    m = np.load('model_data/skanda/' + str(i) + '.npy')
    m_mean = np.average(m, axis=0)
    dist = 1 - np.dot(m_mean, g_mean)/(np.linalg.norm(m_mean)*np.linalg.norm(g_mean))
    sum_dists += dist
    # print(dist)
avg_dist = sum_dists/10
print("[HOST] Avg Dist from Saved Model: %s" % (avg_dist,))

is_correct = avg_dist < 0.17

with open('exp/testutt_guessedspk.txt', 'w') as f:
    f.write("yes" if is_correct else "no")
