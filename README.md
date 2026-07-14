# TORGO ECAPA-TDNN Pooling Experiments

This repository contains a clean WeSpeaker-based experimental pipeline for speaker verification on the TORGO dataset, focusing on the comparison between Attentive Statistics Pooling (ASTP) and Multi-Head Attentive Statistics Pooling (MHASTP) under an ECAPA-TDNN backbone.

## Task

The goal is to evaluate whether multi-head attentive statistical aggregation improves speaker embedding quality for dysarthric speaker verification.

## Dataset Protocol

- Dataset: TORGO
- Microphone: `wav_headMic` only
- Control speakers: `FC01 FC02 FC03 MC01 MC02 MC03 MC04`
- Dysarthric speakers: `F01 F03 F04 M01 M02 M03 M04 M05`
- Split: utterance-level 8:2 train/eval split with fixed random seed
- Evaluation: all-pair utterance-level speaker verification trials

## Main Results

| Method | Control EER | Control minDCF | Dysarthric EER | Dysarthric minDCF |
|---|---:|---:|---:|---:|
| ECAPA-TDNN + ASTP | 13.714 | 0.726 | 14.740 | 0.921 |
| ECAPA-TDNN + MHASTP | 12.628 | 0.677 | 11.131 | 0.842 |

## Main Conclusion

MHASTP improves over ASTP on both the control and dysarthric subsets. The improvement is especially clear on the dysarthric subset, where EER is reduced from 14.740% to 11.131%.

## Environment

```bash
source /data/wangxintong/miniconda3/etc/profile.d/conda.sh
conda activate xintong_wespeaker
source path.sh
```

## Training Examples

### Dysarthric ASTP

```bash
nohup torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  /data/wangxintong/workspace/wespeaker/wespeaker/bin/train.py \
  --config conf/dys_astp.yaml \
  --data_type raw \
  --train_data data/dys_train/raw.list \
  --train_label data/dys_train/utt2spk \
  > logs/dys_astp_train.log 2>&1 &
```

### Dysarthric MHASTP

```bash
nohup torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  /data/wangxintong/workspace/wespeaker/wespeaker/bin/train.py \
  --config conf/dys_mhastp.yaml \
  --data_type raw \
  --train_data data/dys_train/raw.list \
  --train_label data/dys_train/utt2spk \
  > logs/dys_mhastp_train.log 2>&1 &
```

## Evaluation Steps

1. Average last checkpoints.
2. Extract embeddings on eval set.
3. Score all trials with cosine similarity.
4. Compute EER and minDCF.

## Figures

The t-SNE visualization compares dysarthric speaker embeddings extracted by ECAPA-TDNN + ASTP and ECAPA-TDNN + MHASTP.

## Notes

The reported numbers are based on the utterance-level all-pair protocol constructed in this project. They should be compared within the same protocol rather than directly treated as official TORGO benchmark numbers.
