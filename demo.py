# based on mfcc-nns/kaldi-rsr/attention-textind
import numpy as np
import sys
from collections import Counter
from keras.models import load_model
from keras import backend as K
from keras.models import Model

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

    saved_model_path = 'model_data/context_50frms_4mx.model.hdf5'
    model = load_model(saved_model_path)

    assert sys.argv[1]
    utt_data = np.load(sys.argv[1])
    n_frms = utt_data.shape[0] - N_INP_FRMS + 1
    batch_x = np.zeros((n_frms, N_INP_FRMS*MFCC_SIZE))
    for i in range(n_frms):
        batch_x[i] = utt_data[i:(i+N_INP_FRMS), 0:MFCC_SIZE].reshape(1, N_INP_FRMS*MFCC_SIZE)
    nn_output = model.predict_on_batch(batch_x)

    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense_1').output)
    intermediate_output = intermediate_layer_model.predict(batch_x)
    print(intermediate_output.shape)
    spk = 0
    print(score_softmax_outputs(nn_output))
    is_correct = int(spk == score_softmax_outputs(nn_output))

