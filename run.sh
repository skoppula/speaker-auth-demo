#!/bin/bash -e

if [ $1 == "record" ]; then 
    echo "Recording new wav and starting pipeline ..."
    stage=0
elif [ $1 == "process" ]; then
    echo "Processing existing wave file in /tmp/test.wav..."
    stage=2
elif [ $1 == "classify" ]; then
    echo "Classifying existing data in current directory..."
    stage=8
else
    echo "invalid option: argument must be one of record, process, or classify"
    exit 1
fi

nj=1
ubmsize=500
ivectordim=600

export KALDI_ROOT="/home/skoppula/kaldi-trunk"
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin:$KALDI_ROOT/src/ivectorbin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
export LC_ALL=C
export train_cmd=run.pl

for dir in steps utils conf sid 
do
    ln -sf $KALDI_ROOT/egs/sre08/v1/$dir
done

if [ $stage -le 0 ]; then
    echo "Please speak for 5 seconds!..."
    arecord -t wav -d 5 /tmp/test.wav
    echo "Done recording..."
fi

if [ $stage -le 1 ]; then
    echo "Setting up Kaldi files now..."
    mkdir -p data
    echo "test_utt /tmp/test.wav" > data/wav.scp
    echo "test_utt test_spk" > data/utt2spk
    cat data/utt2spk | utils/utt2spk_to_spk2utt.pl > data/spk2utt
    echo "test_spk f" > data/$dataset/spk2gender
    echo "Ending Kaldi files setup now..."
fi

if [ $stage -le 2 ]; then
    echo "Now processing wav files into MFCCs..."
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $nj \
        --cmd "$train_cmd" data exp/mfcc data/mfcc
    echo "Finished MFCCs."
fi

if [ $stage -le 3 ]; then
    echo "Computing VAD..."
    sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
        data exp/vad data/vad
    echo "Finished VAD."
fi

vad_delta_dir="data/dd_mfcc_postvad"

if [ $stage -le 4 ]; then
    echo "Now applying VAD filtering, add deltas, and CMVN filters..."
    mkdir -p $vad_delta_dir
    add-deltas $delta_opts scp:data/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp,s,cs:data/vad.scp ark,scp:$vad_delta_dir/dd_mfcc_postvad.ark,$vad_delta_dir/dd_mfcc_postvad.scp
    echo "Finished applying filters."
fi

#   Extract i-vectors for all of our data.
ie_dir=exp/ie_${ubmsize}_${ivectordim} 
ivec_dir=exp/ivectors_${ubmsize}_${ivectordim}
ivec_norm_dir=exp/ivectors_norm_${ubmsize}_${ivectordim}
mkdir -p $ivec_dir
mkdir -p $ivec_norm_dir

if [ $stage -le 5 ]; then
    echo "Extract i-vectors now..."
    sid/extract_ivectors.sh --cmd "$train_cmd" --nj $nj $ie_dir data $ivec_dir 
    echo "Finished extracting i-vectors."
fi

if [ $stage -le 6 ]; then
    echo "Normalizing i-vectors now..."
    #   Length normalize per-utterance i-vectors for verify
    ivector-normalize-length scp:$ivec_dir/ivector.scp  ark,scp:$ivec_norm_dir/ivector_norm.ark,$ivec_norm_dir/ivector_norm.scp 
    echo "Normalized i-vectors."
fi

if [ $stage -le 7 ]; then
    echo "Converting ark i-vectors to numpy now..."
    #   Length normalize per-utterance i-vectors for verify
    copy-vector scp:$ivec_dir/ivector.scp ark,t:/tmp/ivectors.ark
    python ivec_ark_to_numpy.py /tmp/ivectors.ark ./np/ivecs/
    echo "Finished converting to numpy ."
fi

if [ $stage -le 8 ]; then
    echo "Starting testing neural network classifier..."
    #   Length normalize per-utterance i-vectors for verify
    source activate tensorflow
    python classify.py ./model/curr_best_weights.hdf5 ./model/spk_mappings.pickle np/ivecs/X.npy
    echo "Done classifying speaker."
fi



