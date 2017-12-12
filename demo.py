# based on mfcc-nns/kaldi-rsr/attention-textind
import numpy as np
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

    assert sys.argv[1]
    utt_data = np.load(sys.argv[1])
    n_frms = utt_data.shape[0] - N_INP_FRMS + 1
    batch_x = np.zeros((n_frms, N_INP_FRMS*MFCC_SIZE))
    for i in range(n_frms):
        batch_x[i] = utt_data[i:(i+N_INP_FRMS), 0:MFCC_SIZE].reshape(1, N_INP_FRMS*MFCC_SIZE)

    checkpoint_path = tf.train.latest_checkpoint("./model_data/")
    saver = tf.train.import_meta_graph(checkpoint_path + ".meta", import_scope=None)
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        nn_dvector = sess.run("dense_1/BiasAdd:0", feed_dict={"maxout_dense_1_input:0": batch_x})
        nn_output = sess.run("output_node0:0", feed_dict={"maxout_dense_1_input:0": batch_x})

    print "dvector", nn_dvector.shape
    print "output", nn_output.shape

    spk = 0
    print(score_softmax_outputs(nn_output))
    is_correct = int(spk == score_softmax_outputs(nn_output))

