#!/usr/bin/env bash
#SBATCH --job-name=ctx-prep
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#
# Data preparation for pruned_transducer_stateless7_contextual (CPU only).
# Submit from the icefall root:
#   sbatch egs/librispeech/ASR/pruned_transducer_stateless7_contextual/slurm/prepare.sbatch
#
# Steps (each is skipped if already done):
#   1. link $LIBRISPEECH_DIR into download/, then prepare.sh stages 0-3:
#      manifests, MUSAN (downloaded if missing), fbank (+speed perturbation)
#   2. pretrained stateless7 model + its bpe.model -> data/lang_bpe_500
#   3. biasing word lists from facebookresearch/fbai-speech (is21_deep_bias)

ICEFALL_ROOT=${ICEFALL_ROOT:-$SLURM_SUBMIT_DIR}
source $ICEFALL_ROOT/egs/librispeech/ASR/pruned_transducer_stateless7_contextual/slurm/env.sh

# 1. LibriSpeech + MUSAN -> data/fbank
mkdir -p download
[ -e download/LibriSpeech ] || ln -s $LIBRISPEECH_DIR download/LibriSpeech
[ -e download/musan ] || [ ! -d $MUSAN_DIR ] || ln -s $MUSAN_DIR download/musan
for part in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
  [ -d download/LibriSpeech/$part ] || { echo "Missing $LIBRISPEECH_DIR/$part"; exit 1; }
done
./prepare.sh --stage 0 --stop-stage 3

# 2. Pretrained ASR model. Its bpe.model must be used for training/decoding.
HF=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/resolve/main
mkdir -p $PRETRAINED_DIR/exp $PRETRAINED_DIR/data/lang_bpe_500 data/lang_bpe_500
for f in exp/pretrained.pt data/lang_bpe_500/bpe.model; do
  [ -s $PRETRAINED_DIR/$f ] || curl -fL -o $PRETRAINED_DIR/$f $HF/$f
done
if [ -s data/lang_bpe_500/bpe.model ] && ! cmp -s data/lang_bpe_500/bpe.model $PRETRAINED_DIR/data/lang_bpe_500/bpe.model; then
  echo "data/lang_bpe_500/bpe.model differs from the pretrained model's; not overwriting it."
  echo "Pass --bpe-model $PRETRAINED_DIR/data/lang_bpe_500/bpe.model to train/decode instead."
else
  cp $PRETRAINED_DIR/data/lang_bpe_500/bpe.model data/lang_bpe_500/bpe.model
fi

# 3. Biasing lists (Le et al., Interspeech 2021)
FB=https://raw.githubusercontent.com/facebookresearch/fbai-speech/main/is21_deep_bias
CTX=data/fbai-speech/is21_deep_bias
mkdir -p $CTX/words $CTX/ref
for f in all_rare_words.txt common_words_5k.txt; do
  [ -s $CTX/words/$f ] || curl -fL -o $CTX/words/$f $FB/words/$f
done
for s in test-clean test-other; do
  for n in 100 500 1000 2000; do
    f=$s.biasing_$n.tsv
    [ -s $CTX/ref/$f ] || curl -fL -o $CTX/ref/$f $FB/ref/$f
  done
done

ls -l data/fbank/librispeech_cuts_*.jsonl.gz $CTX/words $CTX/ref $PRETRAINED_DIR/exp
echo "Data preparation done: $(date)"
