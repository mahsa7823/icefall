# Shared settings for the Slurm scripts in this directory.
# Edit this file once for your grid; the *.sbatch scripts source it.

# ---- Python environment (torch + k2 with CUDA, lhotse, icefall deps) ----
# e.g.: source ~/miniforge3/etc/profile.d/conda.sh && conda activate icefall
# e.g.: source ~/venvs/icefall/bin/activate
ENV_SETUP="source ~/venvs/icefall/bin/activate"

# Uncomment/edit if your grid uses environment modules
# module load cuda/12.4

# ---- Paths ----
# Existing raw LibriSpeech directory (the one containing train-clean-100/ etc.)
LIBRISPEECH_DIR=/path/to/LibriSpeech

# Where MUSAN is (or will be downloaded to, ~11 GB); used for noise augmentation
MUSAN_DIR=$ICEFALL_ROOT/egs/librispeech/ASR/download/musan

# Pretrained (frozen) ASR model: icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11
PRETRAINED_DIR=$ICEFALL_ROOT/egs/librispeech/ASR/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11

# Experiment directory (relative to egs/librispeech/ASR)
EXP_DIR=pruned_transducer_stateless7_contextual/exp

# ---- Do not edit below ----
set -eo pipefail
eval "$ENV_SETUP" || { echo "ENV_SETUP failed: $ENV_SETUP"; exit 1; }
export PYTHONPATH=$ICEFALL_ROOT:${PYTHONPATH:-}
cd $ICEFALL_ROOT/egs/librispeech/ASR
set -u
echo "host: $(hostname)  date: $(date)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "python: $(command -v python)"
python -c "import torch, k2; print('torch', torch.__version__, 'k2', k2.__file__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())" || {
  echo "Cannot import torch/k2 with $(command -v python). Check ENV_SETUP in slurm/env.sh."
  exit 1
}
