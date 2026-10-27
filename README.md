# MCPO Reviewer Attachment

## Contents

```text
code/train.py
code/evaluate.py
config/train.json
config/deepspeed.json
data/train.jsonl
data/test.jsonl
requirements.txt
```

`data/train.jsonl` contains 9,877 preference-training samples from MedQA, MedMCQA, and PubMedQA.

`data/test.jsonl` contains the fixed 1,490-question test split.

## Training

```bash
torchrun --nproc_per_node 4 code/train.py \
  --model MODEL \
  --train-file data/train.jsonl \
  --output-dir OUTPUT \
  --deepspeed config/deepspeed.json
```

## Evaluation

```bash
python code/evaluate.py \
  --model MODEL \
  --test-file data/test.jsonl \
  --output-file predictions.jsonl \
  --summary-file metrics.json
```
