# based on mfcc-nns/kaldi-rsr/attention-textind
import numpy as np
from scipy import spatial
import sys
from collections import Counter
import tensorflow as tf

MAX_NUM_FRMS = 0
MFCC_SIZE = 20
N_INP_FRMS = 50
noise_filt = None; sentence_filt = None; 
filters = [noise_filt, sentence_filt]

# assumes inputs in y from 0 ... n
def one_hot_encode_vec(y, num_spks):
    one_hot_lbl = np.zeros(num_spks)
    one_hot_lbl[y] = 1
    return one_hot_lbl

def get_dvector_fn(model):
    return K.function([model.layers[0].input, K.learning_phase()], [model.layers[13].output])

def read_data_lbls(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        return list(map(lambda x: x.split(' ')[0], lines))

def score_softmax_outputs(nn_batch_output):
    max_i = [np.argmax(softmax) for softmax in nn_batch_output]
    counter = Counter(max_i) 
    return counter.most_common(5)[0][0]

if __name__ == '__main__':

    print("Loading network weights...")

    assert sys.argv[1]
    utt_data = np.load(sys.argv[1])
    n_frms = utt_data.shape[0] - N_INP_FRMS + 1
    batch_x = np.zeros((n_frms, N_INP_FRMS*MFCC_SIZE))
    for i in range(n_frms):
        batch_x[i] = utt_data[i:(i+N_INP_FRMS), 0:MFCC_SIZE].reshape(1, N_INP_FRMS*MFCC_SIZE)

    checkpoint_path = tf.train.latest_checkpoint("./model_data/")
    saver = tf.train.import_meta_graph(checkpoint_path + ".meta", import_scope=None)
    print("Running network inference...")
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        nn_dvector = sess.run("dense_1/BiasAdd:0", feed_dict={"maxout_dense_1_input:0": batch_x})
        nn_output = sess.run("output_node0:0", feed_dict={"maxout_dense_1_input:0": batch_x})

    print "dvector", nn_dvector.shape
    np.save('exp/testutt_dvector.npy', nn_dvector)
    print "output", nn_output.shape

    guess_spk = score_softmax_outputs(nn_output)
    print("Guessing speaker", guess_spk)

    g_mean = np.average(nn_dvector, axis=0)
    sum_dists = 0
    for i in range(10):
        m = np.load('model_data/skanda/' + str(i) + '.npy')
        m_mean = np.average(m, axis=0)
        dist = spatial.distance.cosine(m_mean, g_mean)
	sum_dists += dist
        print(dist)
    avg_dist = sum_dists/10
    print('avg dist', avg_dist)

    is_correct = avg_dist < 0.1 

    with open('exp/testutt_guessedspk.txt', 'w') as f:
	f.write('' + str(guess_spk) + ("yes" if is_correct else "no"))

    if is_correct:
    	print("I think its Skanda!")
    else:
	print("I don't think its Skanda!")

