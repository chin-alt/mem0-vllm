# MemReranker-style Pointwise Distillation

This repository is a small-scale, runnable reproduction of the MemReranker
Stage 2 idea: pointwise BCE distillation for domain reranking.

It does not train a large model from scratch. It starts from a Qwen3 reranker
checkpoint, formats each sample as instruction + query + document, and fits
soft teacher labels from your query-doc-score data.

The implementation uses only the Qwen3-Reranker causal LM yes/no-logit scoring
path. It does not generate text:

```text
score = softmax([logit_no, logit_yes])[yes]
```

Training optimizes the equivalent binary logit:

```text
BCEWithLogitsLoss(logit_yes - logit_no, labels / 10)
```

Default base model:

```text
Qwen/Qwen3-Reranker-0.6B
```

The 8-GPU script targets:

```text
Qwen/Qwen3-Reranker-4B
```

## What Is Reproduced

This project focuses on MemReranker Stage 2:

- Student initialized from a Qwen3-Reranker checkpoint.
- Input text is instruction + query + document.
- The model predicts a continuous query-document relevance score in `[0, 1]`.
- Training uses pointwise BCE soft-label distillation.
- Original `labels` are treated as 0-10 scores and normalized as `labels / 10.0`.
- Labels are clipped to `[0, 1]`.
- `reason` is not used for training; it is kept for debugging and case study output.

Differences from the original paper:

- Stage 0 Rank-DistiLLM data training is not reproduced.
- Stage 1 GPT/Qwen ensemble pairwise label generation is not reproduced.
- Stage 3 InfoNCE is not fully reproduced.
- This project uses your existing query-doc-score data as teacher soft labels.

## Data Format

JSONL and JSON arrays are supported.

```json
{
  "instruction": "Score whether the document answers the query.",
  "query": "Which pocket camera I viewed ships faster?",
  "doc": "title: Pocket Camera A, type: product, abstract: Ships tomorrow.",
  "reason": "The document mentions delivery speed.",
  "labels": 9.475
}
```

If `query_id` exists, it is used as the group key. Otherwise, the raw `query`
string is used. Grouping matters for both split leakage prevention and ranking
metrics.

If `doc` is missing, the loader tries to build a document string from fields
such as `title`, `type`, `abstract`, `content`, `text`, and `memory`.

## Install

```bash
pip install -r requirements.txt
```

For QLoRA, use a Linux CUDA environment with `bitsandbytes`.

Training, evaluation, and prediction all use the same Qwen3 causal LM
yes/no-logit path.

If logs say `Can't load the configuration of 'Qwen/Qwen3-Reranker-0.6B'` after
an `httpx.ProxyError` or `504 Gateway Time-out`, the model id is probably fine;
the machine failed to download from Hugging Face. Fix the proxy/mirror or use a
pre-downloaded local model directory.

If logs say a local path like `/home/.../Qwen3-Reranker-0.6B` is an invalid repo
id, the local directory path is wrong or does not exist. Check it on the cluster
first:

```bash
ls -lah /home/c50061497/MemOS/src/memos/reranker/memranker/models/Qwen3-Reranker-0.6B
ls -lah /home/c50061497/MemOS/reranker/memranker/models/Qwen3-Reranker-0.6B
```

Use whichever path actually contains `config.json`, tokenizer files, and model
weight shards. `SWIFT_ATTN_IMPL` is ignored by the paper-aligned evaluator; use
`ATTN_IMPLEMENTATION=eager` or `ATTN_IMPLEMENTATION=flash_attention_2` instead.

Common model ids:

```text
Qwen/Qwen3-Reranker-0.6B
Qwen/Qwen3-Reranker-4B
```

The code also normalizes common typos such as `qwen/qwen3-reranker-0.6` to
`Qwen/Qwen3-Reranker-0.6B`.

## Offline Model Download

On a machine that can reach Hugging Face:

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-Reranker-0.6B \
LOCAL_DIR=models/Qwen3-Reranker-0.6B \
bash scripts/download_qwen3_reranker.sh
```

For the 4B model:

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-Reranker-4B \
LOCAL_DIR=models/Qwen3-Reranker-4B \
bash scripts/download_qwen3_reranker.sh
```

Then copy that directory to the cluster and pass it as a local path:

```bash
python src/evaluate.py \
  --model_path /path/to/Qwen3-Reranker-0.6B \
  --test_file data/split_seed42/test.jsonl \
  --output_dir outputs/baseline_local_model \
  --attn_implementation flash_attention_2
```

All scoring and training loops use `tqdm` progress bars. For non-interactive
logs, add `--disable_tqdm` to `src/train_pointwise.py`.

## Fixed Train/Dev/Test Split

For formal experiments, first export fixed split files with a fixed seed:

```bash
python src/split_data.py \
  --input_file data/all.jsonl \
  --output_dir data/split_seed42 \
  --seed 42 \
  --eval_ratio 0.1 \
  --test_ratio 0.1
```

Or use the helper script:

```bash
INPUT_FILE=data/all.jsonl OUTPUT_DIR=data/split_seed42 bash scripts/split_data.sh
```

This writes:

```text
data/split_seed42/train.jsonl
data/split_seed42/dev.jsonl
data/split_seed42/test.jsonl
data/split_seed42/split_info.json
data/split_seed42/splits.json
```

The split is by query group, so the same `query_id` or same `query` text will
not appear in multiple splits. The original JSON fields are preserved.

## Baseline Evaluation

Evaluate the unfinetuned 0.6B model on the fixed test split:

```bash
TEST_FILE=data/split_seed42/test.jsonl \
OUTPUT_DIR=outputs/baseline_qwen3_reranker_06b \
bash scripts/eval_baseline.sh
```

If the model is already on the machine, set `MODEL_NAME_OR_PATH` to that local
directory:

```bash
TEST_FILE=data/split_seed42/test.jsonl \
MODEL_NAME_OR_PATH=/path/to/Qwen3-Reranker-0.6B \
OUTPUT_DIR=outputs/baseline_qwen3_reranker_06b \
bash scripts/eval_baseline.sh
```

The script defaults to `PRECISION=fp16`, `BATCH_SIZE=16`, and
`ATTN_IMPLEMENTATION=flash_attention_2`. If flash-attn is not installed, set
`ATTN_IMPLEMENTATION=eager`. For a quick throughput check, inspect these fields
in `overall_metrics.json`:

```text
score_time_seconds
seconds_per_example
examples_per_second
```

Common reasons for slow evaluation:

- The run is on CPU, or the model was loaded in fp32 instead of fp16.
- `batch_size=4` and `max_length=4096` are conservative and can underuse the GPU.
- Very long documents make reranking expensive because each query-document pair
  is a full forward pass.
- Hugging Face download/proxy stalls can look like model load latency.

For local 0.6B model evaluation on your Linux machine:

```bash
TEST_FILE=data/split_seed42/test.jsonl \
MODEL_NAME_OR_PATH=/home/c50061497/MemOS/reranker/memranker/models/Qwen3-Reranker-0.6B \
BATCH_SIZE=16 \
MAX_LENGTH=2048 \
ATTN_IMPLEMENTATION=flash_attention_2 \
OUTPUT_DIR=outputs/baseline_qwen3_reranker_06b \
bash scripts/eval_baseline.sh
```

Evaluation outputs:

```text
overall_metrics.json
per_query_metrics.jsonl
predictions.jsonl
```

## Train 0.6B LoRA

```bash
python src/train_pointwise.py \
  --train_file data/split_seed42/train.jsonl \
  --dev_file data/split_seed42/dev.jsonl \
  --test_file data/split_seed42/test.jsonl \
  --output_dir outputs/qwen3_reranker_06b_lora \
  --model_name_or_path Qwen/Qwen3-Reranker-0.6B \
  --max_length 4096 \
  --epochs 3 \
  --lr 2e-5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --attn_implementation flash_attention_2 \
  --use_lora \
  --fp16
```

The script supports automatic model download from Hugging Face when the model
name is used. If your cluster is offline, download the model first and pass the
local path to `--model_name_or_path`.

## ModernBERT BCE Baseline

For an encoder baseline, this repo also supports ModernBERT pointwise BCE
training. It uses the same JSON/JSONL data and the same soft labels:

```text
label = clip(labels / 10.0, 0, 1)
loss  = BinaryCrossEntropyLoss(sequence_classification_logit, label)
score = sigmoid(sequence_classification_logit)
```

This is not the Qwen3 yes/no-logit reranker path. The training script follows
the Sentence-Transformers official reranker style:

```text
CrossEncoder("answerdotai/ModernBERT-base")
CrossEncoderTrainer(...)
BinaryCrossEntropyLoss(...)
```

The input is encoded as a text pair:

```text
text_a = instruction + "\n\nQuery: " + query
text_b = document
```

By default this is a full fine-tune, matching the official CrossEncoder setup:
all ModernBERT parameters are trainable unless the upstream model itself has
frozen parameters. The script logs and writes parameter counts to:

```text
outputs/modernbert_pointwise/parameter_counts.json
outputs/modernbert_pointwise/best/modernbert_reranker_config.json
```

Train with fixed split files:

```bash
TRAIN_FILE=data/split_seed42/train.jsonl \
DEV_FILE=data/split_seed42/dev.jsonl \
TEST_FILE=data/split_seed42/test.jsonl \
MODEL_NAME_OR_PATH=answerdotai/ModernBERT-base \
OUTPUT_DIR=outputs/modernbert_pointwise \
MAX_LENGTH=2048 \
PER_DEVICE_TRAIN_BATCH_SIZE=8 \
ATTN_IMPLEMENTATION=sdpa \
BF16=1 \
bash scripts/train_modernbert_pointwise.sh
```

For multi-GPU training, launch through the same helper by setting
`NUM_PROCESSES`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_PROCESSES=4 \
TRAIN_FILE=data/split_seed42/train.jsonl \
DEV_FILE=data/split_seed42/dev.jsonl \
TEST_FILE=data/split_seed42/test.jsonl \
MODEL_NAME_OR_PATH=answerdotai/ModernBERT-base \
OUTPUT_DIR=outputs/modernbert_pointwise_4gpu \
MAX_LENGTH=2048 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
ATTN_IMPLEMENTATION=sdpa \
BF16=1 \
bash scripts/train_modernbert_pointwise.sh
```

If the model is already downloaded on the cluster, pass the local directory:

```bash
MODEL_NAME_OR_PATH=/path/to/ModernBERT-base \
LOCAL_FILES_ONLY=1 \
bash scripts/train_modernbert_pointwise.sh
```

Evaluate the ModernBERT checkpoint:

```bash
TEST_FILE=data/split_seed42/test.jsonl \
MODEL_PATH=outputs/modernbert_pointwise/best \
OUTPUT_DIR=outputs/modernbert_eval \
BATCH_SIZE=16 \
ATTN_IMPLEMENTATION=sdpa \
BF16=1 \
bash scripts/eval_modernbert.sh
```

Run one-query prediction:

```bash
python src/predict_modernbert.py \
  --model_path outputs/modernbert_pointwise/best \
  --instruction "Judge whether the document is useful for answering the query." \
  --query "Which hotel I viewed is more worry-free for a family trip?" \
  --docs_file data/docs.jsonl \
  --output_file predictions_modernbert_ranked.json \
  --top_k 10 \
  --max_length 2048 \
  --attn_implementation sdpa \
  --bf16
```

ModernBERT supports long-context classification, but memory still grows with
`max_length` and batch size. Start with `MAX_LENGTH=1024` or `2048`, then raise
it only after the baseline is stable.

## Train 4B on 8 RTX 3090 GPUs

Use the 8-GPU helper:

```bash
TRAIN_FILE=data/split_seed42/train.jsonl \
DEV_FILE=data/split_seed42/dev.jsonl \
TEST_FILE=data/split_seed42/test.jsonl \
OUTPUT_DIR=outputs/qwen3_reranker_4b_8x3090_lora \
bash scripts/train_qwen3_reranker_4b_8x3090.sh
```

The script runs:

```text
accelerate launch --num_processes 8 --mixed_precision fp16
```

Important defaults for RTX 3090:

- `--model_name_or_path Qwen/Qwen3-Reranker-4B`
- `--fp16`
- `--use_lora`
- `--gradient_checkpointing`
- `--per_device_train_batch_size 1`
- `--gradient_accumulation_steps 8`
- `--max_length 2048`

The effective batch size is:

```text
num_gpus * per_device_train_batch_size * gradient_accumulation_steps
```

With the defaults, that is `8 * 1 * 8 = 64`.

If memory is still tight, reduce `MAX_LENGTH` to 1024 or add `--load_in_4bit`
to the script command line:

```bash
bash scripts/train_qwen3_reranker_4b_8x3090.sh --load_in_4bit
```

## Train Listwise Soft Labels

Pointwise training treats each query-document pair independently with BCE. The
listwise variant groups all documents for the same query, builds a teacher
distribution from the soft labels, and trains the model distribution over the
documents in that query group:

```text
teacher_probs = softmax((labels / 10) / teacher_temperature)
model_probs   = softmax((logit_yes - logit_no) / model_temperature)
loss          = KL(teacher_probs || model_probs)
```

This is useful when you want the model to learn relative ranking inside each
query list instead of only matching independent scores.

Run 0.6B LoRA listwise training:

```bash
TRAIN_FILE=data/split_seed42/train.jsonl \
DEV_FILE=data/split_seed42/dev.jsonl \
TEST_FILE=data/split_seed42/test.jsonl \
MODEL_NAME_OR_PATH=/path/to/Qwen3-Reranker-0.6B \
OUTPUT_DIR=outputs/qwen3_reranker_06b_listwise_lora \
MAX_LENGTH=2048 \
GROUP_BATCH_SIZE=1 \
GRAD_ACCUM=8 \
bash scripts/train_qwen3_reranker_listwise.sh
```

Run 4B on 8 RTX 3090 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_PROCESSES=8 \
MODEL_NAME_OR_PATH=/home/c50061497/MemOS/src/memos/reranker/memranker/models/Qwen3-Reranker-4B \
OUTPUT_DIR=outputs/qwen3_reranker_4b_8x3090_listwise_lora \
MAX_LENGTH=2048 \
GROUP_BATCH_SIZE=1 \
GRAD_ACCUM=8 \
bash scripts/train_qwen3_reranker_listwise.sh
```

Important listwise knobs:

- `TEACHER_SCORE_SCALE=normalized` uses `labels / 10`; use `raw` for 0-10 labels.
- `TEACHER_TEMPERATURE=1.0` controls how sharp the teacher distribution is.
- `MAX_GROUP_SIZE=16` caps docs per query group for memory; use a smaller value if OOM.
- `GROUP_TRUNCATION=input_order` preserves file order; `label_desc` keeps highest teacher labels first.
- `LOSS_TYPE=kl` matches distribution distillation; `ce` gives the same gradient up to teacher entropy.

Evaluate the saved adapter exactly like pointwise:

```bash
python src/evaluate.py \
  --model_path outputs/qwen3_reranker_4b_8x3090_listwise_lora/best \
  --test_file data/split_seed42/test.jsonl \
  --output_dir outputs/listwise_eval \
  --max_length 2048 \
  --attn_implementation flash_attention_2 \
  --fp16
```

## Train With ms-swift Native Listwise Reranker

This repository also provides an ms-swift route for Qwen3-Reranker listwise
training. It follows the native Swift reranker API:

```text
swift sft --task_type generative_reranker --loss_type listwise_reranker
```

Install Swift dependencies on the Linux training machine:

```bash
pip install -r requirements-swift.txt
```

Version note: this script needs an ms-swift version that exposes
`--task_type generative_reranker` and `--loss_type listwise_reranker`.
ms-swift 2.6.1 is too old for this native reranker path. Its
`swift sft --help` exposes older arguments such as `--model_id_or_path`,
`--sft_type`, `--dtype`, and `--use_flash_attn`, but not the native reranker
task/loss arguments. If you see an error like `ambiguous option: --model could
match --model_type, --model_id_or_path...`, upgrade ms-swift:

```bash
pip uninstall -y ms-swift swift
pip install "ms-swift==3.12.6" --upgrade --upgrade-strategy only-if-needed
swift sft --help | grep -E "task_type|loss_type|train_type|tuner_type|--model "
```

For `torch==2.5.1`, prefer `ms-swift==3.12.6`. ms-swift 4.x imports PyTorch's
newer FSDP2 `FSDPModule`, which is not available in torch 2.5.1. Swift 3.12.x
uses `--train_type lora`; Swift 4.x uses `--tuner_type lora`; Swift 2.x used
`--sft_type lora`. The helper script detects these names automatically.

ms-swift's native listwise loss is not the same as the custom soft-label KL
loss above. Swift expects one query with positive and negative documents, then
optimizes a group cross-entropy objective. The exporter maps your soft labels
to that format:

```text
labels / 10 >= POSITIVE_THRESHOLD  -> positive_messages
labels / 10 <  POSITIVE_THRESHOLD  -> negative_messages
```

If a query has no document above the threshold, the highest-scored document is
used as the positive by default (`POSITIVE_STRATEGY=threshold_or_top1`). The
exported format is:

```json
{
  "messages": [
    {"role": "system", "content": "instruction"},
    {"role": "user", "content": "query"}
  ],
  "positive_messages": [[{"role": "assistant", "content": "relevant doc"}]],
  "negative_messages": [[{"role": "assistant", "content": "irrelevant doc"}]]
}
```

Run 0.6B Swift-native listwise LoRA:

```bash
TRAIN_FILE=data/split_seed42/train.jsonl \
DEV_FILE=data/split_seed42/dev.jsonl \
MODEL_NAME_OR_PATH=/home/c50061497/MemOS/src/memos/reranker/memranker/models/Qwen3-Reranker-0.6B \
OUTPUT_DIR=outputs/qwen3_reranker_06b_swift_listwise_lora \
SWIFT_DATA_DIR=data/swift_listwise_seed42_06b \
MAX_LENGTH=2048 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRAD_ACCUM=8 \
bash scripts/train_qwen3_reranker_swift_listwise.sh
```

Run 4B on 8 RTX 3090 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
TRAIN_FILE=data/split_seed42/train.jsonl \
DEV_FILE=data/split_seed42/dev.jsonl \
MODEL_NAME_OR_PATH=/home/c50061497/MemOS/src/memos/reranker/memranker/models/Qwen3-Reranker-4B \
OUTPUT_DIR=outputs/qwen3_reranker_4b_8x3090_swift_listwise_lora \
SWIFT_DATA_DIR=data/swift_listwise_seed42_4b \
MAX_LENGTH=2048 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRAD_ACCUM=8 \
DEEPSPEED=zero2 \
TORCH_DTYPE=float16 \
ATTN_IMPL=flash_attn \
bash scripts/train_qwen3_reranker_swift_listwise.sh
```

Useful Swift-native knobs:

- `POSITIVE_THRESHOLD=0.7` controls soft-score to positive/negative conversion.
- `MAX_POSITIVE_SAMPLES=1` and `MAX_NEGATIVE_SAMPLES=7` are consumed by ms-swift during listwise group construction.
- `LISTWISE_RERANKER_TEMPERATURE=1.0` controls Swift's listwise softmax temperature.
- `ATTN_IMPL=flash_attn` plus `PADDING_FREE=true` is the recommended fast path; if flash-attn is unavailable, set `ATTN_IMPL=eager PADDING_FREE=false`.
- `USE_HF=true` makes Swift download from Hugging Face instead of ModelScope; local model paths work without network.

The wrapper first writes Swift JSONL files under `SWIFT_DATA_DIR`, then starts
`swift sft`. Swift checkpoints are written under `OUTPUT_DIR`, usually as a
versioned directory containing `checkpoint-*`. You can evaluate a Swift LoRA
checkpoint with the existing evaluator if the checkpoint contains PEFT
`adapter_config.json`:

```bash
python src/evaluate.py \
  --model_path outputs/qwen3_reranker_4b_8x3090_swift_listwise_lora/vx-xxx/checkpoint-xxx \
  --test_file data/split_seed42/test.jsonl \
  --output_dir outputs/swift_listwise_eval \
  --max_length 2048 \
  --attn_implementation flash_attention_2 \
  --fp16
```

Or merge LoRA with Swift before deployment:

```bash
swift export \
  --adapters outputs/qwen3_reranker_4b_8x3090_swift_listwise_lora/vx-xxx/checkpoint-xxx \
  --merge_lora true
```

## Finetuned Evaluation

Use the same fixed test split:

```bash
python src/evaluate.py \
  --model_path outputs/qwen3_reranker_4b_8x3090_lora/best \
  --test_file data/split_seed42/test.jsonl \
  --output_dir outputs/finetuned_eval \
  --max_length 2048 \
  --attn_implementation flash_attention_2 \
  --fp16
```

Compare:

```text
outputs/baseline_qwen3_reranker_06b/overall_metrics.json
outputs/finetuned_eval/overall_metrics.json
```

Main metrics:

- BCE
- MSE
- Pearson
- Spearman
- MAP
- MRR
- NDCG@1
- NDCG@3
- NDCG@10
- Recall@1
- Recall@3
- Recall@5

NDCG uses the normalized graded label in `[0, 1]`. MAP, MRR, and Recall use a
binary relevance threshold. The default is normalized label `>= 0.7`, equivalent
to original label `>= 7`.

## Business Evaluation

For business recall data, use `src/evaluate_business.py`. It builds the same
reranker input used in training:

```text
<Instruct>: {instruction}
<Query>: {query}
<Document>: {doc}
```

The model layer then wraps that block with the Qwen3-Reranker chat prefix and
scores the final `yes/no` logits. Do not pass ad-hoc text such as
`query: ... document: ...` directly if you want scores to match training.

Ground truth is an Excel or CSV file. Defaults:

```text
query column: query
doc id column: PageId
```

`PageId` can contain multiple correct IDs in one cell, separated by Chinese or
English commas, for example:

```text
dy_PDP_32，dy_PDP_33，dy_PDP_31，dy_PDP_34
```

Recall JSON may be either:

```json
{
  "winter down jacket": [
    {"id": "page_1", "text": "productName: ..."},
    {"id": "page_2", "text": "productName: ..."}
  ]
}
```

or a list of rows with `query` and `docs`/`documents`.

Run:

```bash
GT_FILE=data/business/ground_truth.xlsx \
RECALL_FILE=data/business/recall.json \
MODEL_PATH=outputs/qwen3_reranker_06b_lora/best \
OUTPUT_DIR=outputs/business_eval \
MAX_LENGTH=2048 \
BATCH_SIZE=16 \
ATTN_IMPLEMENTATION=flash_attention_2 \
bash scripts/eval_business.sh
```

If your Excel columns or recall JSON keys differ:

```bash
python src/evaluate_business.py \
  --gt_file data/business/ground_truth.xlsx \
  --recall_file data/business/recall.json \
  --gt_query_col query \
  --gt_doc_id_col PageId \
  --gt_sheet Sheet1 \
  --recall_id_key id \
  --recall_text_key text \
  --model_path outputs/qwen3_reranker_06b_lora/best \
  --output_dir outputs/business_eval \
  --max_length 2048 \
  --batch_size 16 \
  --attn_implementation flash_attention_2 \
  --fp16
```

Outputs:

```text
metrics.json
per_query_metrics.jsonl
predictions.jsonl
business_eval.xlsx
business_eval.csv
```

For each query, if the ground truth has `N` correct IDs, the script takes the
model's top-`N` reranked IDs and computes:

```text
accuracy = number of hit IDs / N
```

The xlsx summary keeps the original `query` and `PageId`, plus model recalled
IDs, model scores, hit IDs, missing IDs, per-query accuracy, and estimated
inference time. `predictions.jsonl` also stores every reranked document with its
raw model `score`.

The script also supports a dynamic score-based cutoff requested for business
evaluation. For each query, scores are min-max normalized, prefix sums are
computed, and the cutoff is selected by maximizing:

```text
ExpectedFbeta@k = (1 + beta^2) * cumulative_gain@k / (beta^2 * total_gain + k)
```

The default is `beta=0.3`, configurable through:

```bash
EXPECTED_FBETA_BETA=0.3 bash scripts/eval_business.sh
```

Per-query outputs include `BestK@ExpectedFbeta`, `ExpectedFbeta@BestK`,
`BestK截断ID`, `BestK截断ID和分数`, `Precision@BestK`, `Recall@BestK`, and
`F1@BestK`. Matrix summary tables also include the averaged BestK metrics.
Business metrics are averaged over all ground-truth queries, so a query with no
matched recalled docs contributes zero instead of silently disappearing.

To recompute the dynamic cutoff and real F1 for multiple beta values from
existing `predictions.jsonl` files without rerunning model inference:

```bash
OUTPUT_ROOT=outputs/business_matrix_xxx \
bash scripts/recompute_beta_f1.sh
```

The default beta list is:

```text
1.0 0.7 0.5 0.3 0.2
```

Outputs are written under `OUTPUT_ROOT`:

```text
beta_f1_matrix.xlsx
beta_f1_matrix_summary.csv
beta_f1_matrix_summary.json
beta_f1_matrix_per_query.jsonl
```

Each run directory also gets `beta_f1.xlsx`, `beta_f1_summary.csv`, and
`beta_f1_per_query.csv`. The summary columns follow the mentor-style report:
`avg_selected_count`, `avg_precision`, `avg_recall`, and `avg_f1`.

### Business Evaluation Matrix

To compare the three business datasets across the five local models requested
for the latency-delay experiment, run:

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_LENGTH=2048 \
BATCH_SIZE=4 \
EXPECTED_FBETA_BETA=0.3 \
PRECISION=fp16 \
ATTN_IMPLEMENTATION=flash_attention_2 \
bash scripts/eval_business_matrix.sh
```

The matrix script evaluates:

```text
models/IAAR-Shanghai/MemReranker-4B
models/Qwen3-Reranker-0.6B
models/Qwen3-Reranker-4B
outputs/qwen3_reranker_4b_8x3090_lora/best
outputs/qwen3_reranker_06b_lora/best
```

against:

```text
data/latency_delay/0428caption
data/latency_delay/0428keyword
data/latency_delay/0625caption
```

The first two datasets use `--gt_doc_id_col PageId_new`; `0625caption` uses the
default `PageId`. Each dataset/model pair gets its own output directory and
also a uniquely named copy under `OUTPUT_ROOT`. The final comparison tables are:

```text
summary_metrics.xlsx
summary_metrics.csv
summary_metrics.json
```

If a long matrix run is interrupted, rerun the same command with the same
`OUTPUT_ROOT`; by default `SKIP_EXISTING=1` skips pairs that already have
`metrics.json`. You can rebuild only the summary table with:

```bash
python src/summarize_business_matrix.py --output_root outputs/business_matrix_xxx
```

The matrix script launches one Python process per dataset/model pair, so GPU
memory is released between runs when that process exits. `evaluate_business.py`
also clears the CUDA cache after writing outputs. If a 4B run still OOMs, lower
`BATCH_SIZE` first, then `MAX_LENGTH`; if memory is plentiful, raise
`BATCH_SIZE` to improve throughput.

`metrics.json` and matrix summary tables also record two PyTorch GPU peak
memory fields for each run:

```text
cuda_peak_allocated_mib       PyTorch tensor peak memory for this process
cuda_peak_reserved_mib        PyTorch caching allocator peak reservation
```

These fields describe the current Python process. `cuda_peak_reserved_mib` is
usually the cleaner headline number for "how much GPU memory this run needed".

### CrossEncoder Business Matrix

For ModernBERT or mBERT rerankers trained with the Sentence-Transformers
`CrossEncoderTrainer`, use the CrossEncoder business matrix entry instead of
the Qwen yes/no-logit matrix script:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODERNBERT_MODEL_PATH=/home/c50061497/MemOS/src/memos/reranker/memranker/outputs/modernbert_pointwise/best \
MBERT_MODEL_PATH=/home/c50061497/MemOS/src/memos/reranker/memranker/outputs/mbert_pointwise/best \
MAX_LENGTH=2048 \
BATCH_SIZE=32 \
PRECISION=bf16 \
ATTN_IMPLEMENTATION=sdpa \
SCORE_ACTIVATION=sigmoid \
bash scripts/eval_business_matrix_crossencoder.sh
```

This script evaluates the same three business datasets:

```text
0428caption   gt_doc_id_col=PageId_new
0428keyword   gt_doc_id_col=PageId_new
0625caption   gt_doc_id_col=PageId
```

The two default model names are `modernbert` and `mbert`. To run an arbitrary
set of CrossEncoder checkpoints, pass matching model names and paths:

```bash
MODEL_NAMES="modernbert mbert another_run" \
MODEL_PATHS="/path/to/modernbert/best|/path/to/mbert/best|/path/to/another/best" \
bash scripts/eval_business_matrix_crossencoder.sh
```

Outputs follow the same layout as the Qwen business matrix:

```text
<OUTPUT_ROOT>/<dataset>__<model>/metrics.json
<OUTPUT_ROOT>/<dataset>__<model>/business_eval.xlsx
<OUTPUT_ROOT>/summary_metrics.xlsx
<OUTPUT_ROOT>/summary_metrics.csv
<OUTPUT_ROOT>/summary_metrics.json
```

`SCORE_ACTIVATION=sigmoid` is the default and matches pointwise BCE soft-label
training. Use `identity` only if you intentionally want raw CrossEncoder logits.

### vLLM Business Evaluation

For faster offline scoring with Qwen3-Reranker sequence-classification support,
use the independent vLLM evaluator. It keeps the same ground-truth parsing,
recall parsing, metrics, and output filenames as `src/evaluate_business.py`, but
replaces the local Transformers scorer with `vllm==0.10.2`
`LLM(..., runner="pooling").score()`.

The vLLM evaluator now has two scoring backends:

```text
SCORING_BACKEND=pooling    fast path; uses Qwen3ForSequenceClassification + classifier_from_token
SCORING_BACKEND=generate   official Qwen3-Reranker path; generates one yes/no token and reads logprobs
```

Use `generate` to rule out model-class or scoring-head problems. It is slower,
but it does not depend on the pooling runner or sequence-classification override.

Install vLLM dependencies on the Linux GPU machine:

```bash
pip install -r requirements-vllm.txt
```

For a CUDA 12.8 machine with torch 2.8.0, a clean environment can be prepared as:

```bash
uv pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements-vllm.txt --torch-backend=cu128
```

If vLLM fails during tokenizer initialization with
`Qwen2Tokenizer has no attribute all_special_tokens_extended`, the environment
has an incompatible Transformers stack. `vllm==0.10.2` should use the
Transformers 4.x stack, not Transformers 5.x. Reinstall the vLLM-side
dependencies in the dedicated eval environment:

```bash
uv pip install --upgrade --force-reinstall \
  "transformers>=4.55.2,<5.0.0" \
  "tokenizers>=0.21.1,<0.22.0" \
  "sentencepiece>=0.2.0"
```

Some local Qwen tokenizer configs store `extra_special_tokens` as a list, while
`transformers==4.55.x` expects a dict. If you see
`AttributeError: 'list' object has no attribute 'keys'`, the evaluator now
creates a patched tokenizer copy under `<output_dir>/_vllm_tokenizer` and passes
that copy to vLLM. The original model directory is not modified.

Single run:

```bash
python business_eval_vllm.py \
  --gt_file data/business/ground_truth.xlsx \
  --recall_file data/business/recall.json \
  --model_path /home/c50061497/MemOS/src/memos/reranker/memranker/models/Qwen3-Reranker-4B \
  --output_dir outputs/business_eval_vllm_2048_bs256 \
  --max_length 2048 \
  --batch_size 256 \
  --scoring_backend generate \
  --dtype bfloat16 \
  --gpu_memory_utilization 0.90 \
  --tensor_parallel_size 1 \
  --max_num_batched_tokens 32768 \
  --max_num_seqs 256 \
  --sort_by_length
```

Run the same 5-model x 3-dataset matrix with vLLM:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MAX_LENGTH=2048 \
BATCH_SIZE=64 \
DTYPE=float16 \
GPU_MEMORY_UTILIZATION=0.80 \
TENSOR_PARALLEL_SIZE=2 \
MAX_NUM_BATCHED_TOKENS=8192 \
MAX_NUM_SEQS=64 \
bash scripts/eval_business_matrix_vllm.sh
```

For matrix experiments, set the same environment variable:

```bash
SCORING_BACKEND=generate bash scripts/eval_business_matrix_vllm.sh
```

The vLLM matrix writes the same per-run files plus
`summary_metrics.{xlsx,csv,json}`.

vLLM expects `--model_path` to be a full Hugging Face model directory with
`config.json`. A PEFT/LoRA adapter directory such as
`outputs/qwen3_reranker_4b_8x3090_lora/best` only has `adapter_config.json`, so
merge it once before running the vLLM evaluator:

```bash
python src/merge_lora.py \
  --adapter_path outputs/qwen3_reranker_4b_8x3090_lora/best \
  --base_model_path models/Qwen3-Reranker-4B \
  --output_dir outputs/qwen3_reranker_4b_8x3090_lora_merged \
  --torch_dtype float16 \
  --overwrite

python src/merge_lora.py \
  --adapter_path outputs/qwen3_reranker_06b_lora/best \
  --base_model_path models/Qwen3-Reranker-0.6B \
  --output_dir outputs/qwen3_reranker_06b_lora_merged \
  --torch_dtype float16 \
  --overwrite
```

The vLLM matrix uses those merged paths by default. To use custom merged paths:

```bash
QWEN3_RERANKER_4B_LORA_PATH=/path/to/qwen3_4b_lora_merged \
QWEN3_RERANKER_06B_LORA_PATH=/path/to/qwen3_06b_lora_merged \
bash scripts/eval_business_matrix_vllm.sh
```

### Ascend 910B4 vLLM Business Evaluation

For Huawei Ascend 910B4 / Atlas A2 inference, there are two local inference
paths. Use local vLLM-Ascend first when the dependency stack can be installed;
use plain `torch-npu + transformers` as the fallback when vLLM-Ascend is blocked
by the cloud image or Python package mirror. Both paths run on the 910B4 host
itself. They do not use an HTTP service, remote API, or external inference
provider. The business matrix logic stays the same: ground-truth parsing,
recall parsing, Qwen3-Reranker prompt formatting, `predictions.jsonl`,
`business_eval.{csv,xlsx}`, and `summary_metrics.{csv,json,xlsx}` are shared
across the evaluators.

The unified local entrypoint defaults to vLLM-Ascend:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
bash scripts/install_ascend_vllm_910b4.sh

ASCEND_RT_VISIBLE_DEVICES=0 \
DATA_ROOT=/path/to/data/latency_delay \
MODEL_ROOT=/path/to/Qwen3-Reranker-4B \
OUTPUTS_ROOT=/path/to/outputs \
BACKEND=vllm \
DTYPE=float16 \
BATCH_SIZE=4 \
MAX_LENGTH=2048 \
bash scripts/eval_business_matrix_ascend_local.sh
```

When `MODEL_ROOT` points directly to a Hugging Face model directory containing
`config.json`, the Ascend vLLM script evaluates only that model. You can also
be explicit:

```bash
MODEL_NAME=qwen3_reranker_4b \
MODEL_PATH=/path/to/Qwen3-Reranker-4B \
bash scripts/eval_business_matrix_ascend_vllm.sh
```

If vLLM-Ascend still cannot be installed, switch only the backend and
requirements. This fallback runs the Qwen3-Reranker CausalLM prompt locally on
NPU and reads the final yes/no logits, matching the official Transformers
reranker practice:

```bash
pip install -r requirements-ascend-torch.txt \
  -i https://mirrors.huaweicloud.com/repository/pypi/simple

ASCEND_RT_VISIBLE_DEVICES=0 \
DATA_ROOT=/path/to/data/latency_delay \
MODEL_ROOT=/path/to/models \
OUTPUTS_ROOT=/path/to/outputs \
BACKEND=torch \
DEVICE=npu \
PRECISION=fp16 \
ATTN_IMPLEMENTATION=eager \
BATCH_SIZE=1 \
MAX_LENGTH=2048 \
bash scripts/eval_business_matrix_ascend_local.sh
```

For the torch fallback, start with `BATCH_SIZE=1` for 4B checkpoints. If a
single pair scores successfully and memory is stable, increase it to `2` or
`4`. Use `ATTN_IMPLEMENTATION=eager` as the safest first run on torch-npu; try
`sdpa` only after the eager path is verified on your image.

Prepare the Ascend runtime on the target machine. The closest public vLLM-Ascend
line that works with Huawei Cloud mirrors is `0.11.0rc1`:

```text
Python       >=3.9,<3.12
CANN/NNAL    8.3.RC1 or the matching cloud-provided 8.3 package
torch        2.7.1
torch-npu    2.7.1
vLLM         0.11.0
vllm-ascend  0.11.0rc1
numpy        <2.0.0
transformers 4.55.2
tokenizers   0.21.4
```

The original CUDA environment has `torch==2.8.0`, `vllm==0.10.2`,
`numpy==2.2.6`, `xformers`, `triton`, and many `nvidia-*` packages. Do not
reuse that full lockfile on Ascend; keep the business/data packages, add
`torch-npu`/`vllm-ascend`, and remove the CUDA-only packages. The Ascend
requirements file keeps the original versions where they are compatible. On
Python 3.10+ this includes
`transformers==4.55.2`, `tokenizers==0.21.4`, `accelerate==1.14.0`,
`peft==0.19.1`, `pandas==2.3.3`, `openpyxl==3.1.5`, and `tqdm==4.68.4`.
Do not install `vllm-ascend==0.10.2rc1` from the public mirrors: that wheel
requires the unpublished/dev package `torch-npu==2.7.1.dev20250724`, so pip
cannot resolve it on restricted Huawei Cloud images.
The default `requirements-ascend-vllm.txt` targets aarch64 Ascend cloud hosts
and avoids `download.pytorch.org`, because many NPU clouds only allow the
Huawei Cloud PyPI mirrors. For x86_64 NPU hosts, use
`requirements-ascend-vllm-x86_64.txt`.

For a manual environment, source CANN and NNAL before installing the Python
stack:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
bash scripts/install_ascend_vllm_910b4.sh
```

Python 3.9 is supported by the published vLLM 0.11.0 metadata and by the
`vllm-ascend` cp39 wheel. The vLLM 0.11.0 source accidentally contains three
files with Python 3.10-style union annotations. The installer runs
`src/vllm_py39_compat.py` after package installation to replace only those
known annotations with `Optional`/`Union`; it is idempotent and keeps `.orig`
backups next to changed files. The evaluation entrypoint applies the same
checked patch automatically before importing vLLM.

The requirements use Python markers to preserve the original package versions
on Python 3.10+, while Python 3.9 receives the last compatible releases for
`accelerate`, `peft`, `datasets`, `scikit-learn`, `scipy`, `Pillow`, `numba`,
`llvmlite`, and `ray`. These packages cannot keep the newer pins on Python 3.9
because their package metadata excludes that interpreter.

Do not install `torch`, `torch-npu`, `vllm`, and `vllm-ascend` in one pip
command. The public `vllm==0.11.0` wheel metadata depends on `torch==2.8.0`,
while vLLM-Ascend `0.11.0rc1` runs with `torch/torch-npu==2.7.1`; the install
script handles this by installing vLLM packages with `--no-deps` after the
Ascend torch stack is present.

If the cloud cannot reach even the Huawei mirrors, build a wheelhouse on a
networked aarch64 Linux machine or a matching container, copy it to the NPU
host, and install offline:

```bash
pip download -r requirements-ascend-vllm.txt \
  -d wheelhouse \
  -i https://mirrors.huaweicloud.com/repository/pypi/simple
tar -czf wheelhouse-ascend-vllm-aarch64.tar.gz wheelhouse

# On the restricted cloud host:
tar -xzf wheelhouse-ascend-vllm-aarch64.tar.gz
pip install --no-index --find-links wheelhouse -r requirements-ascend-vllm.txt
```

If your cloud image exposes `vllm-ascend==0.11.0` final but not `0.11.0rc1`,
use `vllm==0.11.0` with `vllm-ascend==0.11.0`; keep
`torch==2.7.1` and `torch-npu==2.7.1`.

If building `vllm-ascend` from source on a CPU-only build host, set the chip
target before installation. Prefer building on the target 910B4 host when
possible so `npu-smi` can detect the SoC automatically; otherwise use the
SoC name supported by your `vllm-ascend` branch and CANN package:

```bash
export SOC_VERSION=ascend910b1  # Atlas A2 example from vLLM-Ascend docs
```

Run the 5-model x 3-dataset matrix on visible NPU chips:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
DATA_ROOT=/path/to/data/latency_delay \
MODEL_ROOT=/path/to/models \
OUTPUTS_ROOT=/path/to/outputs \
MAX_LENGTH=2048 \
BATCH_SIZE=64 \
DTYPE=float16 \
TENSOR_PARALLEL_SIZE=2 \
MAX_NUM_BATCHED_TOKENS=8192 \
MAX_NUM_SEQS=64 \
bash scripts/eval_business_matrix_ascend_vllm.sh
```

The Ascend script defaults to local/offline model loading and `pooling`
scoring, using the same Qwen3-Reranker `hf_overrides` as the vLLM Ascend
Qwen3-Reranker guide:

```text
architectures=["Qwen3ForSequenceClassification"]
classifier_from_token=["no", "yes"]
is_original_qwen3_reranker=true
```

For W8A8 Ascend-converted weights, pass the vLLM quantization name:

```bash
VLLM_QUANTIZATION=ascend bash scripts/eval_business_matrix_ascend_vllm.sh
```

For extra vLLM Ascend tuning, pass JSON through `VLLM_ADDITIONAL_CONFIG`:

```bash
VLLM_ADDITIONAL_CONFIG='{"enable_flashcomm1": true}' \
bash scripts/eval_business_matrix_ascend_vllm.sh
```

Do not use `CUDA_VISIBLE_DEVICES`, `flash_attention_2`, or `bitsandbytes` for
the Ascend run. If you need the slower Transformers fallback for debugging,
install `torch-npu`, set `ASCEND_RT_VISIBLE_DEVICES`, and run
`src/evaluate_business.py` with `ATTN_IMPLEMENTATION=sdpa` or `eager`.

## C-MTEB Retrieval Generalization

C-MTEB Retrieval is a group of Hugging Face datasets rather than one local
file. The verified layout uses separate `corpus` and `queries` parquet splits
under repos such as `C-MTEB/T2Retrieval`, `C-MTEB/MMarcoRetrieval`,
`C-MTEB/DuRetrieval`, `C-MTEB/CovidRetrieval`, `C-MTEB/CmedqaRetrieval`,
`C-MTEB/EcomRetrieval`, and `C-MTEB/MedicalRetrieval`.

Install the conversion dependencies:

```bash
pip install -r requirements-cmteb.txt
```

Download the retrieval subsets:

```bash
OUTPUT_DIR=data/cmteb_r/raw \
DATASETS="T2Retrieval MMarcoRetrieval DuRetrieval CovidRetrieval CmedqaRetrieval EcomRetrieval MedicalRetrieval" \
INCLUDE_QRELS=1 \
bash scripts/download_cmteb_r.sh
```

If you manually download parquet files from Hugging Face, keep either of these
layouts:

```text
data/cmteb_r/raw/T2Retrieval/data/corpus-*.parquet
data/cmteb_r/raw/T2Retrieval/data/queries-*.parquet
data/cmteb_r/raw/T2Retrieval-qrels/data/*.parquet
```

or:

```text
data/cmteb_r/raw/T2Retrieval/corpus-*.parquet
data/cmteb_r/raw/T2Retrieval/queries-*.parquet
```

You can also pass `--input_dir` directly to `.../T2Retrieval` or
`.../T2Retrieval/data` for a single manually downloaded dataset.
Keep the paired qrels repo next to it as `T2Retrieval-qrels`, or pass
`QRELS_INPUT_DIR=/path/to/qrels/base` when preparing the JSONL.

Inspect local columns and supervision coverage before conversion:

```bash
python src/inspect_cmteb_r.py \
  --input_dir data/cmteb_r/raw \
  --datasets T2Retrieval \
  --qrels_input_dir data/cmteb_r/raw \
  --output_file data/cmteb_r/t2_inspect.json
```

C-MTEB qrels repos use rows like `qid`, `pid`, `score`. The inspection output
reports `qrels_files`, `qrels_query_count`, and `recommended_supervision_strategy`.
For `T2Retrieval`, the expected qrels source is typically
`C-MTEB/T2Retrieval-qrels`.

Convert them to the same MemReranker JSONL format used by `src/evaluate.py`.
The converter keeps the original query group, writes `doc_id`, assigns positive
documents label `10.0`, and samples random corpus negatives with label `0.0`:

```bash
INPUT_DIR=data/cmteb_r/raw \
QRELS_INPUT_DIR=data/cmteb_r/raw \
OUTPUT_FILE=data/cmteb_r/cmteb_r_eval.jsonl \
DATASETS="T2Retrieval" \
NEGATIVES_PER_QUERY=15 \
MAX_QUERIES_PER_DATASET=1000 \
MAX_DOCS_PER_QUERY=32 \
SEED=42 \
SUPERVISION_STRATEGY=explicit \
bash scripts/prepare_cmteb_r_eval.sh
```

It also writes `data/cmteb_r/cmteb_r_eval.metadata.json`, including discovered
columns, exported query counts, and qrels/positive coverage. If a subset exports
zero records, inspect that metadata first; some mirrors may omit qrels or store
positive ids in a nonstandard field. To continue while diagnosing such subsets,
add `SKIP_MISSING_QRELS=1` to the prepare command.

Important: `corpus-*.parquet` plus `queries-*.parquet` only proves that documents
and queries are readable. It is not enough for NDCG/MAP/Recall unless supervision
can be recovered from qrels, positive document fields, or a verified fallback
rule. Prefer the paired `*-qrels` repo whenever it exists. `inspect_cmteb_r.py`
reports `has_supervision`; if it is `false`, you need an additional qrels or
ground-truth file.

Evaluate with the regular Transformers evaluator:

```bash
python src/evaluate.py \
  --test_file data/cmteb_r/cmteb_r_eval.jsonl \
  --model_path /path/to/Qwen3-Reranker-0.6B \
  --output_dir outputs/cmteb_r_eval_06b \
  --max_length 2048 \
  --batch_size 8 \
  --fp16
```

Evaluate faster with the vLLM scorer:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TEST_FILE=data/cmteb_r/cmteb_r_eval.jsonl \
MODEL_PATH=/path/to/Qwen3-Reranker-4B \
OUTPUT_DIR=outputs/cmteb_r_vllm_4b \
MAX_LENGTH=2048 \
BATCH_SIZE=64 \
SCORING_BACKEND=generate \
DTYPE=float16 \
TENSOR_PARALLEL_SIZE=2 \
MAX_NUM_BATCHED_TOKENS=8192 \
MAX_NUM_SEQS=64 \
bash scripts/eval_cmteb_r_vllm.sh
```

`scripts/eval_cmteb_r_vllm.sh` reads the model from `MODEL_PATH`. Variables such
as `MODEL_NAME_OR_PATH` are not used by this script. The script defaults to
`LOCAL_FILES_ONLY=1`, so a typo in a local path fails immediately instead of
falling back to a Hugging Face download attempt. To allow a remote HF id, set
`LOCAL_FILES_ONLY=0`.

For inference-debug runs, compare `SCORING_BACKEND=pooling` with
`SCORING_BACKEND=generate` on the same candidate file. If Qwen3-Reranker stays
high but MemReranker only drops under `pooling`, suspect the vLLM pooling
model-class/scoring-head path. If MemReranker is low under both backends, the
drop is more likely from fine-tuning/data/domain effects than from score
extraction.

If the progress bar appears to get slower over time, check the `max_chars` and
`sec` postfix printed for each vLLM batch. With `--sort_by_length`, shorter
pairs are scored first and longer pairs are scored later, so later batches can
naturally be slower. Use `SORT_DESCENDING=1` to score long batches first, or
`--no_sort_by_length` for unsorted batches.

Both evaluators write:

```text
overall_metrics.json
per_query_metrics.jsonl
predictions.jsonl
```

This random-negative conversion is a practical reranker generalization sanity
test. For a stricter retrieval benchmark, use a fixed first-stage retriever run
as the candidate set and score those candidates with the same evaluator.

For a business-like CMTEB-R reranker evaluation, build a first-stage candidate
list from the corpus first, then score only those candidates with the reranker.
The default builder uses `Qwen/Qwen3-Embedding-0.6B` to retrieve top-100
candidates and uses qrels only for labels/metrics. It does not force missed
positives into the candidate set, so first-stage recall loss remains visible:

```bash
INPUT_DIR=data/cmteb_r \
QRELS_INPUT_DIR=data/cmteb_r \
OUTPUT_FILE=data/cmteb_r/cmteb_r_qwen3_embedding_candidates.jsonl \
DATASETS="T2Retrieval" \
RETRIEVAL_BACKEND=embedding \
EMBEDDING_MODEL_NAME_OR_PATH=/path/to/Qwen3-Embedding-0.6B \
EMBEDDING_LOCAL_FILES_ONLY=1 \
EMBEDDING_CACHE_DIR=data/cmteb_r/embedding_cache \
EMBEDDING_SEARCH_DEVICE=cuda \
CANDIDATE_TOP_K=100 \
MAX_QUERIES_PER_DATASET=1000 \
INDEX_DOC_MAX_CHARS=2048 \
bash scripts/build_cmteb_r_candidates.sh
```

If you want the script to download/cache the embedding model from Hugging Face,
use `EMBEDDING_MODEL_NAME_OR_PATH=Qwen/Qwen3-Embedding-0.6B` and omit
`EMBEDDING_LOCAL_FILES_ONLY=1`. To run the older lexical baseline instead, set
`RETRIEVAL_BACKEND=bm25`.

For faster candidate building on multi-GPU machines, let sentence-transformers
encode corpus/query embeddings with one worker per GPU and keep the top-k search
on GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
INPUT_DIR=data/cmteb_r \
QRELS_INPUT_DIR=data/cmteb_r \
OUTPUT_FILE=data/cmteb_r/cmteb_r_qwen3_embedding_candidates.jsonl \
DATASETS="T2Retrieval" \
RETRIEVAL_BACKEND=embedding \
EMBEDDING_MODEL_NAME_OR_PATH=/path/to/Qwen3-Embedding-0.6B \
EMBEDDING_LOCAL_FILES_ONLY=1 \
EMBEDDING_MULTI_PROCESS=1 \
EMBEDDING_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
EMBEDDING_BATCH_SIZE=64 \
EMBEDDING_CHUNK_SIZE=2000 \
EMBEDDING_SEARCH_DEVICE=cuda:0 \
EMBEDDING_SEARCH_DTYPE=float16 \
CANDIDATE_TOP_K=100 \
MAX_QUERIES_PER_DATASET=1000 \
bash scripts/build_cmteb_r_candidates.sh
```

The first run still needs to encode the corpus; later runs reuse
`EMBEDDING_CACHE_DIR`, so changing only `MAX_QUERIES_PER_DATASET` or the reranker
model should be much faster. If GPU memory is tight, reduce
`EMBEDDING_BATCH_SIZE` or set `EMBEDDING_SEARCH_DEVICE=cpu`.

Then rerank that candidate file:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TEST_FILE=data/cmteb_r/cmteb_r_qwen3_embedding_candidates.jsonl \
MODEL_PATH=/path/to/Qwen3-Reranker-4B \
OUTPUT_DIR=outputs/cmteb_r_qwen3_embedding_vllm_4b \
MAX_LENGTH=2048 \
BATCH_SIZE=64 \
TENSOR_PARALLEL_SIZE=auto \
EXPECTED_FBETA_BETAS="0.2 0.3 0.5 0.7 1.0" \
bash scripts/eval_cmteb_r_vllm.sh
```

`TENSOR_PARALLEL_SIZE=auto` counts `CUDA_VISIBLE_DEVICES`, so the command above
passes `--tensor_parallel_size 4` to vLLM. If you want a single-card run while
multiple GPUs are visible, set `TENSOR_PARALLEL_SIZE=1`.

For higher throughput when each GPU can hold one full reranker copy, use the
data-parallel evaluator instead. It splits the JSONL by query group, launches
one vLLM worker per GPU with `tensor_parallel_size=1`, then merges predictions
and recomputes global metrics:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DEVICES=0,1,2,3 \
NUM_SHARDS=4 \
TEST_FILE=data/cmteb_r/cmteb_r_qwen3_embedding_candidates.jsonl \
MODEL_PATH=/path/to/Qwen3-Reranker-4B \
OUTPUT_DIR=outputs/cmteb_r_qwen3_embedding_vllm_4b_dp \
MAX_LENGTH=2048 \
BATCH_SIZE=64 \
EXPECTED_FBETA_BETAS="0.2 0.3 0.5 0.7 1.0" \
bash scripts/eval_cmteb_r_vllm_dp.sh
```

The data-parallel runner shows one aggregate progress bar in the main terminal.
Set `SHOW_PROGRESS=0` to hide it, or tune refresh frequency with
`PROGRESS_POLL_INTERVAL=0.5`.

Data-parallel shard logs and intermediate outputs are kept under
`OUTPUT_DIR/_dp_shards/shard_*/`. The merged root `OUTPUT_DIR` contains the same
files as the single-process evaluator, plus `shard_metrics.jsonl`.

If long-tail documents make later batches too slow, prepare a 10% fast subset
first. This script now filters overlong query-document pairs before sampling,
then samples query groups until the output is close to 10% of the original row
count:

```bash
INPUT_FILE=data/cmteb_r/cmteb_r_qwen3_embedding_candidates.jsonl \
OUTPUT_FILE=data/cmteb_r/cmteb_r_qwen3_embedding_candidates_10pct_maxlen2048.jsonl \
SAMPLE_RATIO=0.10 \
SEED=42 \
MAX_LENGTH=2048 \
MAX_DOCS_PER_QUERY=0 \
bash scripts/prepare_fast_eval_subset.sh
```

Then point the evaluator to the subset:

```bash
TEST_FILE=data/cmteb_r/cmteb_r_qwen3_embedding_candidates_10pct_maxlen2048.jsonl \
OUTPUT_DIR=outputs/cmteb_r_qwen3_embedding_vllm_4b_dp_10pct \
bash scripts/eval_cmteb_r_vllm_dp.sh
```

Rows whose formatted instruction+query+doc character length is greater than
`MAX_LENGTH` are dropped before sampling. Groups with no remaining relevant docs
are dropped by default. Sampling is done by query group, not by individual row,
so each sampled query keeps its candidate list and ranking metrics remain
meaningful. Set `MAX_DOC_CHARS` only if you still want post-filter truncation,
or `MAX_DOCS_PER_QUERY` to cap the candidate count per query.

This writes the normal ranking outputs plus dynamic-cutoff reports:

```text
beta_f1_summary.csv
beta_f1_summary.json
beta_f1_per_query.jsonl
```

`IdealTopK` is the number of qrels-positive documents present in the first-stage
candidate list for that query. The evaluator also reports
`Precision@IdealTopK`, `Recall@IdealTopK`, and `F1@IdealTopK` by truncating the
model-ranked list at that ideal candidate count. `CandidateRecall` uses the full
qrels positive count as denominator, so first-stage misses and length-filter
misses are visible in the final report.

### CMTEB-R CrossEncoder Evaluation

For ModernBERT or mBERT checkpoints trained with `CrossEncoderTrainer`, use the
CrossEncoder JSONL evaluator instead of the Qwen/vLLM evaluator:

```bash
CUDA_VISIBLE_DEVICES=0 \
TEST_FILE=data/cmteb_r/cmteb_r_qwen3_embedding_candidates.jsonl \
MODERNBERT_MODEL_PATH=outputs/modernbert_pointwise/best \
MBERT_MODEL_PATH=outputs/mbert_pointwise/best \
MAX_LENGTH=2048 \
BATCH_SIZE=32 \
PRECISION=bf16 \
ATTN_IMPLEMENTATION=sdpa \
SCORE_ACTIVATION=sigmoid \
EXPECTED_FBETA_BETAS="0.2 0.3 0.5 0.7 1.0" \
bash scripts/eval_cmteb_r_crossencoder.sh
```

The script evaluates both default model names, `modernbert` and `mbert`, and
writes one run directory per model plus a summary table:

```text
outputs/cmteb_r_crossencoder_<timestamp>/cmteb_r__modernbert/overall_metrics.json
outputs/cmteb_r_crossencoder_<timestamp>/cmteb_r__mbert/overall_metrics.json
outputs/cmteb_r_crossencoder_<timestamp>/summary_metrics.xlsx
```

To run arbitrary CrossEncoder checkpoints:

```bash
MODEL_NAMES="modernbert mbert another_run" \
MODEL_PATHS="/path/to/modernbert/best|/path/to/mbert/best|/path/to/another/best" \
bash scripts/eval_cmteb_r_crossencoder.sh
```

`SCORE_ACTIVATION=sigmoid` matches pointwise BCE soft-label training. Use
`identity` only if you intentionally want raw CrossEncoder logits.

For four-card or eight-card data-parallel CrossEncoder evaluation, run one
full model copy per GPU and split the JSONL by query group:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DEVICES=0,1,2,3 \
NUM_SHARDS=4 \
TEST_FILE=data/cmteb_r/cmteb_r_qwen3_embedding_candidates.jsonl \
MODEL_PATH=/path/to/modernbert_or_mbert_checkpoint \
OUTPUT_DIR=outputs/cmteb_r_crossencoder_dp_modernbert \
MAX_LENGTH=8192 \
BATCH_SIZE=32 \
PRECISION=bf16 \
SCORE_ACTIVATION=sigmoid \
EXPECTED_FBETA_BETAS="0.2 0.3 0.5 0.7 1.0" \
bash scripts/eval_cmteb_r_crossencoder_dp.sh
```

For eight cards, use `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`,
`DEVICES=0,1,2,3,4,5,6,7`, and `NUM_SHARDS=8`. This is data parallel:
each GPU loads one model and scores a different subset of queries. It does
not split one model across GPUs, so lower `BATCH_SIZE` first if a single
ModernBERT/mBERT copy does not fit.

The CMTEB-R candidate JSONL has positives if it was built with qrels, for
example through `build_cmteb_r_candidates.py` with `QRELS_INPUT_DIR` set.
Positive qrels documents are written with positive labels, negative candidates
with negative labels, and the evaluator treats `label >= 0.7` as relevant by
default. The reranker still only sorts the first-stage candidate list; if the
Qwen3-Embedding top-k candidate list misses all qrels positives for a query,
`CandidateRecall`/`candidate_relevant_count` will expose that retrieval miss.

## LoCoMo Reranker Evaluation

LoCoMo `locomo10.json` is a long-term conversation QA benchmark. For reranker
evaluation, do not use `answer` as the document. The correct retrieval unit is
the conversation turn referenced by `qa[].evidence`:

```text
query = qa[].question
positive doc ids = qa[].evidence, for example D1:3
candidate docs = all dialogue turns from the same sample.conversation
doc text = session time + speaker + turn text + optional BLIP image caption
```

The converter below builds a realistic two-stage setup: first retrieve top-100
candidate turns inside each conversation with Qwen3-Embedding, then label the
candidates whose `dia_id` appears in `evidence` as positive. Missed evidence is
not appended by default, so `CandidateRecall` tells you how many gold evidence
turns survived first-stage retrieval.

```bash
INPUT_FILE=data/locomo/locomo10.json \
OUTPUT_FILE=data/locomo/locomo_qwen3_embedding_candidates.jsonl \
RETRIEVAL_BACKEND=embedding \
EMBEDDING_MODEL_NAME_OR_PATH=/home/c50061497/MemOS/src/memos/reranker/memranker/models/Qwen3-Embedding-0.6B \
EMBEDDING_LOCAL_FILES_ONLY=1 \
CANDIDATE_TOP_K=100 \
bash scripts/prepare_locomo_reranker.sh
```

For a quick parser-only smoke test without the embedding model, use BM25:

```bash
INPUT_FILE=data/locomo/locomo10.json \
OUTPUT_FILE=data/locomo/locomo_bm25_candidates.jsonl \
RETRIEVAL_BACKEND=bm25 \
CANDIDATE_TOP_K=100 \
bash scripts/prepare_locomo_reranker.sh
```

The output JSONL is compatible with the same vLLM/DP evaluator used by CMTEB-R:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DEVICES=0,1,2,3 \
NUM_SHARDS=4 \
TEST_FILE=data/locomo/locomo_qwen3_embedding_candidates.jsonl \
MODEL_PATH=/home/c50061497/MemOS/src/memos/reranker/memranker/models/IAAR-Shanghai/MemReranker-4B \
OUTPUT_DIR=outputs/locomo_memreranker_4b_vllm_dp \
MAX_LENGTH=2048 \
BATCH_SIZE=64 \
EXPECTED_FBETA_BETAS="0.2 0.3 0.5 0.7 1.0" \
bash scripts/eval_cmteb_r_vllm_dp.sh
```

Useful converter switches:

```text
ENSURE_POSITIVES=1          append missed evidence turns for oracle-style reranker-only analysis
MAX_SAMPLES=1               debug one LoCoMo conversation
MAX_QUERIES=20              debug first 20 QA groups
INCLUDE_ANSWER_METADATA=1   keep answer fields as metadata only, never as model input
```

For pure reranker quality, build an oracle-style candidate file with
`ENSURE_POSITIVES=1`; otherwise LoCoMo metrics include first-stage retrieval
misses from Qwen3-Embedding/BM25. If `CandidateRecall` is low, NDCG/MRR are
capped even when the reranker itself is working.

After evaluation, analyze LoCoMo by query/category and export bad cases:

```bash
EVAL_DIR=outputs/locomo_memreranker_4b_vllm_dp \
TEST_FILE=data/locomo/locomo_qwen3_embedding_candidates.jsonl \
BAD_NDCG_THRESHOLD=0.5 \
BAD_RANK_THRESHOLD=10 \
TOP_K=10 \
bash scripts/analyze_locomo_results.sh
```

For older evaluation outputs, keep `TEST_FILE` set so the analyzer can join back
LoCoMo metadata such as `category`, `evidence`, and `sample_id`. New evaluator
outputs preserve these fields directly in `predictions.jsonl`.

The analyzer writes:

```text
locomo_analysis/locomo_overall_summary.json
locomo_analysis/locomo_category_summary.csv
locomo_analysis/locomo_query_analysis.csv
locomo_analysis/locomo_bad_cases.jsonl
locomo_analysis/locomo_bad_cases.csv
locomo_analysis/locomo_bad_case_report.md
```

`locomo_query_analysis.csv` has one row per QA query, including category,
candidate recall, NDCG/Recall/MRR, the first relevant rank, evidence ids, and
the model's top-1 document. `locomo_bad_cases.jsonl` keeps top-ranked document
snippets and positive document snippets for diagnosis. A query is marked bad
when first-stage retrieval misses all evidence, `NDCG@BAD_NDCG_K` is below
`BAD_NDCG_THRESHOLD`, or the first positive rank is greater than
`BAD_RANK_THRESHOLD`.

### LoCoMo CrossEncoder Evaluation

The same ModernBERT/mBERT CrossEncoder checkpoints can be evaluated on LoCoMo
candidate JSONL with automatic category and bad-case analysis:

```bash
CUDA_VISIBLE_DEVICES=0 \
TEST_FILE=data/locomo/locomo_qwen3_embedding_candidates.jsonl \
MODERNBERT_MODEL_PATH=outputs/modernbert_pointwise/best \
MBERT_MODEL_PATH=outputs/mbert_pointwise/best \
MAX_LENGTH=2048 \
BATCH_SIZE=32 \
PRECISION=bf16 \
ATTN_IMPLEMENTATION=sdpa \
SCORE_ACTIVATION=sigmoid \
TOP_K=10 \
BAD_NDCG_THRESHOLD=0.5 \
BAD_RANK_THRESHOLD=10 \
bash scripts/eval_locomo_crossencoder.sh
```

Each model output directory contains the standard JSONL ranking outputs and the
LoCoMo analysis folder:

```text
outputs/locomo_crossencoder_<timestamp>/locomo__modernbert/overall_metrics.json
outputs/locomo_crossencoder_<timestamp>/locomo__modernbert/predictions.jsonl
outputs/locomo_crossencoder_<timestamp>/locomo__modernbert/locomo_diagnostics.json
outputs/locomo_crossencoder_<timestamp>/locomo__modernbert/locomo_analysis/
outputs/locomo_crossencoder_<timestamp>/summary_metrics.xlsx
```

The summary table includes ranking metrics, dynamic beta metrics, inference
latency, throughput, and CUDA peak memory fields.

If LoCoMo `NDCG@10` is unexpectedly low, first open
`locomo_diagnostics.json`. The most important fields are
`candidate_summary.candidate_recall`,
`candidate_summary.groups_without_candidate_positive`,
`prediction_summary.model_NDCG@10`,
`prediction_summary.inverted_NDCG@10`, and
`prediction_summary.positive_minus_negative_score_mean`.

## Prediction

Prepare `docs.jsonl`:

```json
{"doc": "title: PocketCam A, abstract: Ships tomorrow."}
{"title": "PocketCam B", "abstract": "Ships in two weeks."}
```

Run:

```bash
python src/predict.py \
  --model_path outputs/qwen3_reranker_4b_8x3090_lora/best \
  --instruction "Score whether the document answers the query." \
  --query "Which pocket camera ships faster?" \
  --docs_file docs.jsonl \
  --top_k 10 \
  --output_file predictions_ranked.json \
  --attn_implementation flash_attention_2 \
  --fp16
```

## Smoke Test

The repository supports a `--mock` scorer for local pipeline checks without a
GPU or downloaded model:

```bash
python src/evaluate.py --test_file tmp/smoke/toy.jsonl --output_dir tmp/smoke/eval --mock
python src/predict.py --instruction "rank" --query "fast pocket camera delivery" --docs_file tmp/smoke/docs.jsonl --output_file tmp/smoke/predictions_ranked.json --mock
```

The mock scorer is only for smoke tests. Do not use it for real experiments.
