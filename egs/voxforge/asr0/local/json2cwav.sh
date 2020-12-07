#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration

debugmode=1
dumpdir=./dump   # directory to dump full features
verbose=1      # verbose option

# json folders
test_set="et_en"
train_set="tr_en"
dev_set="dt_en"

# data
datadir=./downloads

# exp tag
tag="" # tag for managing experiments.

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

echo "Scatter data stage 0: Initialising.. "

if [ ! -d ${datadir} ]; then
  echo Cannot find data root! Exiting...
  exit 1
fi

SECONDS=0
a=$SECONDS
diff=0
for x in $test_set $dev_set $train_set; do
  python local/json2json1.py $x ${dumpdir}/${x}/deltafalse/scat.json
  diff=$(( SECONDS - a ))
  echo "$(($diff / 3600)) hours, $((($diff / 60) % 60)) minutes and $(($diff % 60)) seconds elapsed."
done

diff=$(( SECONDS - a ))
echo "Completed in $(($diff / 3600)) hours, $((($diff / 60) % 60)) minutes and $(($diff % 60)) seconds."

#for x in test train; do
#    for f in text wav.scp utt2spk; do
#        sort data/${x}/${f} -o data/${x}/${f}
#    done
#    utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
#done

#feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
#feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
#if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#    ### Task dependent. You have to design training and dev sets by yourself.
#    ### But you can utilize Kaldi recipes in most cases
#    echo "stage 1: Feature Generation"
#    fbankdir=fbank
#    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
#    for x in test train; do
#        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 8 --write_utt2num_frames true \
#            data/${x} exp/make_fbank/${x} ${fbankdir}
#        utils/fix_data_dir.sh data/${x}
#    done
#
#    # make a dev set
#    utils/subset_data_dir.sh --first data/train 100 data/${train_dev}
#    n=$(($(wc -l < data/train/text) - 100))
#    utils/subset_data_dir.sh --last data/train ${n} data/${train_set}
#
#    # compute global CMVN
#    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
#
#    # dump features
#    dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
#        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
#    dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
#        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
#    for rtask in ${recog_set}; do
#        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
#        dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
#            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
#            ${feat_recog_dir}
#    done
#fi
#
#dict=data/lang_1char/${train_set}_units.txt
#echo "dictionary: ${dict}"
