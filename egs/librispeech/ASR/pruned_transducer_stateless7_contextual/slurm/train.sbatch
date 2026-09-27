#!/usr/bin/env bash
#SBATCH --job-name=ctx-train
##SBATCH --gres=gpu:4
#SBATCH --gpus=2
##SBATCH --nodes=4
##SBATCH --cpus-per-task=16
##SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=gpu-a100   #queue
#SBATCH --account=a100acct


module purge
module load conda
conda activate /export/fs04/a12/mahsay/miniconda3/envs/icefall

#
# Train the biasing modules on top of the frozen pretrained stateless7 model
# (full LibriSpeech, 4 GPUs requested; uses however many are allocated). Submit from the icefall root:
#   sbatch egs/librispeech/ASR/pruned_transducer_stateless7_contextual/slurm/train.sbatch
#
# To resume, resubmit with START_EPOCH set to the next epoch, e.g.
#   START_EPOCH=5 sbatch .../train.sbatch
# (epoch-4.pt must exist in EXP_DIR). --init-asr-ckpt is only used when
# START_EPOCH=1.

ICEFALL_ROOT=${ICEFALL_ROOT:-$SLURM_SUBMIT_DIR}
source $ICEFALL_ROOT/egs/librispeech/ASR/pruned_transducer_stateless7_contextual/slurm/env.sh

START_EPOCH=${START_EPOCH:-1}
NUM_EPOCHS=${NUM_EPOCHS:-10}
MAX_DURATION=${MAX_DURATION:-1600}  # per GPU; lower it on OOM
N_DISTRACTORS=${N_DISTRACTORS:-100}
MASTER_PORT=${MASTER_PORT:-$((12000 + RANDOM % 1000))}
# Use as many GPUs as Slurm actually gave this job
NUM_GPUS=${NUM_GPUS:-$(python -c "import torch; print(torch.cuda.device_count())")}
[ "$NUM_GPUS" -ge 1 ] || { echo "No GPU visible to this job"; exit 1; }
echo "Training on $NUM_GPUS GPU(s), --max-duration $MAX_DURATION per GPU"

mkdir -p $EXP_DIR
python pruned_transducer_stateless7_contextual/train.py \
  --world-size $NUM_GPUS \
  --master-port $MASTER_PORT \
  --use-fp16 true \
  --full-libri true \
  --max-duration $MAX_DURATION \
  --exp-dir $EXP_DIR \
  --bpe-model data/lang_bpe_500/bpe.model \
  --init-asr-ckpt $PRETRAINED_DIR/exp/pretrained.pt \
  --context-dir data/fbai-speech/is21_deep_bias \
  --n-distractors $N_DISTRACTORS \
  --start-epoch $START_EPOCH \
  --num-epochs $NUM_EPOCHS \
  --prune-range 5

echo "Training done: $(date)"
