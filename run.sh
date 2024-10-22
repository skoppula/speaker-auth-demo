#!/bin/bash -e
# source activate pi

rm -f /tmp/xilinx.log
touch /tmp/xilinx.log
cat /dev/ttyUSB1 > /tmp/xilinx.log &
echo "s" > /dev/ttyUSB1

if [ $1 == "record" ]; then 
    echo "[HOST] Recording new wav and starting pipeline ..."
    stage=0
elif [ $1 == "process" ]; then
    echo "[HOST] Processing existing wave file in ./tmp/test.wav..."
    stage=2
elif [ $1 == "classify" ]; then
    echo "[HOST] Classifying existing data in current directory..."
    stage=6
else
    echo "[HOST] Invalid option: argument must be one of record, process, or classify"
    exit 1
fi

nj=1
ubmsize=500
ivectordim=600

export KALDI_ROOT="/home/pi/kaldi-trunk"
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin:$KALDI_ROOT/src/ivectorbin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
export LC_ALL=C
export train_cmd=run.pl

for dir in steps utils conf sid 
do
    ln -sf $KALDI_ROOT/egs/sre08/v1/$dir
done

if [ $stage -le 0 ]; then
    rm -f ./tmp/test.wav
    echo "[HOST] Please speak for 5 seconds!..."
    arecord -t wav -d 5 ./tmp/test.wav -f S16_LE
    echo "[HOST] Done recording..."
fi

echo '[HOST] Sending audio samples to ZYNC FPGA for processing...'
echo "d" > /dev/ttyUSB1

if [ $stage -le 1 ]; then
    mkdir -p data
    echo "test_utt ./tmp/test.wav" > data/wav.scp
    echo "test_utt test_spk" > data/utt2spk
    cat data/utt2spk | utils/utt2spk_to_spk2utt.pl > data/spk2utt
    echo "test_spk f" > data/$dataset/spk2gender
fi

if [ $stage -le 2 ]; then
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $nj \
        --cmd "$train_cmd" data exp/mfcc data/mfcc > /dev/null 2>&1
fi

if [ $stage -le 3 ]; then
    sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
        data exp/vad data/vad > /dev/null 2>&1
fi

vad_delta_dir="data/dd_mfcc_postvad"

if [ $stage -le 4 ]; then
    mkdir -p $vad_delta_dir
    add-deltas $delta_opts scp:data/feats.scp ark:tmp/tmp1.ark > /dev/null 2>&1
    apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:tmp/tmp1.ark ark:tmp/tmp2.ark > /dev/null 2>&1
    select-voiced-frames ark:tmp/tmp2.ark scp,s,cs:data/vad.scp ark,scp:$vad_delta_dir/dd_mfcc_postvad.ark,$vad_delta_dir/dd_mfcc_postvad.scp > /dev/null 2>&1
fi

if [ $stage -le 5 ]; then
    python scripts/read_mfcc_ark.py
fi

cat exp/test_utt.txt > /dev/ttyUSB1
sleep 4 # FPGA processing
cat /tmp/xilinx.log
if [ -s /tmp/xilinx.log ]
then
	echo "[HOST] Did not receive response from FPGA."
else
	echo "[HOST] Received FPGA response."
	if [ $stage -le 6 ]; then
	    # python demoV2.py exp/test_utt.npy
	    python process_fpga_response.py /tmp/xilinx.log
	fi


	if [ $stage -le 7 ]; then
		if [[ $(cat exp/testutt_guessedspk.txt | grep "yes") ]]; then
			echo "[HOST] I think it's Skanda. Passed verification!"
		else
			echo "[HOST] I dont think it's Skanda. Did not pass verification!"
		fi
	fi

fi

kill $(ps aux | grep '[c]at /dev/ttyUSB1' | awk '{print $2}')


