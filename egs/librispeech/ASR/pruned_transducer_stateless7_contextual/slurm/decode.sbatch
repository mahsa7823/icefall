#!/usr/bin/env bash
#SBATCH --job-name=ctx-decode
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#
# Decode test-clean/test-other with the predefined biasing lists of
# facebookresearch/fbai-speech and report WER, U-WER (unbiased words) and
# B-WER (biased words). Submit from the icefall root:
#   EPOCH=10 AVG=3 sbatch egs/librispeech/ASR/pruned_transducer_stateless7_contextual/slurm/decode.sbatch
#
# Also decodes a no-biasing baseline with the same checkpoint.
# Results: $EXP_DIR/decode-epoch-$EPOCH-avg-$AVG.txt (WER, U-WER, B-WER per run)
#          $EXP_DIR/modified_beam_search/{wer-summary,recogs,errs,log-decode}-*

ICEFALL_ROOT=${ICEFALL_ROOT:-$SLURM_SUBMIT_DIR}
source $ICEFALL_ROOT/egs/librispeech/ASR/pruned_transducer_stateless7_contextual/slurm/env.sh

EPOCH=${EPOCH:?set EPOCH, e.g. EPOCH=10}
AVG=${AVG:-1}
USE_AVG_MODEL=${USE_AVG_MODEL:-true}  # needs epoch-$((EPOCH-AVG)).pt when true
BIASING_LIST_SIZES=${BIASING_LIST_SIZES:-"100 500 1000 2000"}

RESULTS=$EXP_DIR/decode-epoch-$EPOCH-avg-$AVG.txt

decode() {
  echo "=== decode $*" | tee -a $RESULTS
  python pruned_transducer_stateless7_contextual/decode.py \
    --epoch $EPOCH \
    --avg $AVG \
    --use-averaged-model $USE_AVG_MODEL \
    --exp-dir $EXP_DIR \
    --bpe-model data/lang_bpe_500/bpe.model \
    --context-dir data/fbai-speech/is21_deep_bias \
    --is-predefined true \
    --decoding-method modified_beam_search \
    --beam-size 4 \
    --max-duration 600 \
    "$@" 2>&1 | tee -a $RESULTS
}

# Baseline: same checkpoint, all biasing off
decode --n-distractors 100 \
  --no-encoder-biasing true --no-decoder-biasing true --no-wfst-lm-biasing true

# Neural biasing with biasing lists of different sizes
for n in $BIASING_LIST_SIZES; do
  decode --n-distractors $n \
    --no-encoder-biasing false --no-decoder-biasing false --no-wfst-lm-biasing true
done

echo "Summary (WER(U-WER/B-WER) for test-clean, then test-other):"
grep -E "^=== decode|^[0-9.]+\([0-9.]+/[0-9.]+\)$" $RESULTS
echo "Decoding done: $(date)"
