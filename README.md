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

For a cloud host whose driver is fixed at `24.1.rc2.3` (HDK 24.1.RC2), use
the older official compatibility intersection instead of the default 0.11
stack:

```text
CANN/NNAL    8.1.RC1
Python       3.9-3.11
torch        2.5.1
torch-npu    2.5.1
vLLM         0.8.5.post1
vllm-ascend  0.8.5rc1
transformers 4.51.3
tokenizers   0.21.1
numpy        1.26.4
```

Install that profile with:

```bash
# Install the matching CANN user-space packages first; do not reinstall the
# host driver, firmware, or MCU:
#   Ascend-cann-toolkit_8.1.RC1_linux-aarch64.run
#   Ascend-cann-kernels-910b_8.1.RC1_linux-aarch64.run
#   Ascend-cann-nnal_8.1.RC1_linux-aarch64.run
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
bash scripts/install_ascend_vllm_hdk24rc2.sh
```

Verify package versions, CANN/NNAL visibility, and a real NPU tensor operation:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
python scripts/verify_ascend_vllm_hdk24rc2.py
```

Pass a local model path to also initialize vLLM and complete one generation:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
python scripts/verify_ascend_vllm_hdk24rc2.py \
  --model-path /path/to/Qwen3-Reranker-4B
```

Qwen3 generation is supported on this vLLM-Ascend line, but native
Qwen3-Reranker pooling was added later. Keep inference local and score the
model's yes/no token probabilities through the generation runner:

```bash
SCORING_BACKEND=generate \
BATCH_SIZE=1 \
MAX_NUM_SEQS=1 \
bash scripts/eval_business_matrix_ascend_vllm.sh
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
packages whose newer releases dropped Python 3.9. The selected 3.9 versions
still satisfy vLLM 0.11.0's minimum constraints and provide aarch64 wheels;
Python 3.10+ retains the original newer pins.

This stack pins `setuptools==80.9.0` because setuptools 82 removed the legacy
`pkg_resources` module that some dependencies still import. To repair an
existing environment without reinstalling the full stack, run:

```bash
python -m pip install --force-reinstall "setuptools==80.9.0" \
  -i https://mirrors.huaweicloud.com/repository/pypi/simple
python -c "import pkg_resources; print('pkg_resources ok')"
```

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

### Atlas 300I / 310P container

The vLLM Ascend project identifies `v0.10.0rc1-310p` as the stable
experimental image for Atlas 300I Duo. If startup fails in
`torch.npu.graph(...).capture_begin()` with
`ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE`, first disable ACL Graph capture. This
keeps inference on the NPU and only switches vLLM to eager execution:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
MODEL_NAME=qwen3_reranker_4b \
MODEL_PATH=/path/to/Qwen3-Reranker-4B \
DATA_ROOT=/path/to/data/latency_delay \
OUTPUT_ROOT=/path/to/output \
SCORING_BACKEND=generate \
DTYPE=float16 \
MAX_LENGTH=8192 \
BATCH_SIZE=32 \
MAX_NUM_BATCHED_TOKENS=8192 \
MAX_NUM_SEQS=32 \
ENFORCE_EAGER=1 \
VLLM_COMPILATION_CONFIG='{"custom_ops":["none","+rms_norm","+rotary_embedding"]}' \
bash scripts/eval_business_matrix_ascend_vllm.sh
```

Check the container stack and the host driver mounted into it before running:

```bash
python - <<'PY'
import sys
from importlib.metadata import version
import torch
import torch_npu

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
for package in ("vllm", "vllm-ascend"):
    print(f"{package}:", version(package))
print("device:", torch.npu.get_device_name(0))
PY
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
cat /usr/local/Ascend/driver/version.info
npu-smi info -t board -i 0
```

The `v0.10.0rc1-310p` image uses the CANN 8.2.RC1 generation of the userspace
stack. Docker still uses the host NPU driver and firmware. On 310P this release
does not support ACL Graph, so keep `ENFORCE_EAGER=1` and `DTYPE=float16`. Use
`SCORING_BACKEND=generate` for Qwen3-Reranker: it performs the official local
yes/no-token scoring path through vLLM. Qwen3-Reranker pooling is currently
listed for A2/A3, not 310P, and must not be assumed to be correct on this image.
Only treat the warning as a general driver-stack incompatibility if eager
generation also fails.

Do not use `CUDA_VISIBLE_DEVICES`, `flash_attention_2`, or `bitsandbytes` for
the Ascend run. If you need the slower Transformers fallback for debugging,
install `torch-npu`, set `ASCEND_RT_VISIBLE_DEVICES`, and run
`src/evaluate_business.py` with `ATTN_IMPLEMENTATION=sdpa` or `eager`.

#### Experimental low-latency Qwen3-Reranker path on HDK 24.1.RC2.x

For `Qwen3-Reranker-0.6B`, try the sequence-classification pooling path before
quantization. It replaces one-token generation plus full-vocabulary logprobs
with the two-token yes/no classification head. The launcher keeps the host
driver unchanged, uses the `v0.10.2rc1-310p` userspace image, applies only the
decoder pooling fixes, runs a runtime preflight, warms up the model, and records
batch/pair p50 and p95 latency:

```bash
HOST_REPO_PATH=/home/reranker_experiment/mem0-vllm \
HOST_DATA_PATH=/home/reranker_experiment/data/latency_delay \
HOST_MODEL_PATH=/home/reranker_experiment/model/Qwen3-Reranker-0.6B \
HOST_OUTPUT_PATH=/home/reranker_experiment/output/qwen3_pooling_fp16 \
DATASET=0428caption \
SCORING_BACKEND=pooling \
MAX_LENGTH=1024 \
BATCH_SIZE=16 \
MAX_NUM_BATCHED_TOKENS=4096 \
WARMUP_PAIRS=16 \
PULL_IMAGE=0 \
bash scripts/run_qwen3_reranker_vllm_310p_container.sh
```

The patch is version checked and reversible. For Qwen3 it skips the unrelated
GTE encoder-only attention backport:

```bash
python scripts/patch_vllm_ascend_0102_310p.py --decoder-pooling-only
python scripts/patch_vllm_ascend_0102_310p.py --restore
```

Run an FP16 generate-versus-pooling sweep with production data:

```bash
HOST_MODEL_PATH=/home/reranker_experiment/model/Qwen3-Reranker-0.6B \
HOST_OUTPUT_BASE=/home/reranker_experiment/output/qwen3_310p_ab \
DATASET=0428caption \
BACKENDS="generate pooling" \
BATCH_SIZES="1 8 16 32" \
MAX_LENGTHS="512 1024" \
PULL_IMAGE=0 \
bash scripts/benchmark_qwen3_reranker_310p.sh
```

Each `metrics.json` now contains `batch_latency_p50_seconds`,
`batch_latency_p95_seconds`, `pair_latency_p50_seconds`, and
`pair_latency_p95_seconds` (the last two are amortized batch time per pair).
The sweep also writes `benchmark_summary.json`.

Static W8A8 on this legacy stack remains experimental. Export only static
per-tensor-activation/per-channel-weight W8A8 with a ModelSlim release whose
`quant_model_description.json` is compatible with vLLM-Ascend 0.10.2. Keep
`embed_tokens` and `lm_head` as `FLOAT`; vLLM creates the converted `score`
head as unquantized FP32. Do not use dynamic/per-token W8A8. Before loading the
model, run the read-only preflight:

The repository includes an end-to-end exporter for the production training
JSONL. It clones and installs the vLLM-compatible ModelSlim ref/tag into a
dedicated virtual environment, creates `inputs_pretokenized` calibration rows,
truncates only the document so the Qwen3 answer-position suffix is retained,
exports conservative static W8A8 weights, and validates the result. It never
installs or changes the host driver:

```bash
cd /home/reranker_experiment/mem0-vllm

TRAIN_JSONL=/home/reranker_experiment/data/split/train.jsonl \
FLOAT_MODEL_PATH=/home/reranker_experiment/model/Qwen3-Reranker-0.6B \
QUANT_MODEL_PATH=/home/reranker_experiment/model/Qwen3-Reranker-0.6B-W8A8-static-safe \
MAX_LENGTH=1024 \
CALIB_SAMPLES=64 \
CALIB_BACKEND=pooling \
bash scripts/quantize_qwen3_reranker_w8a8_static_310p.sh
```

By default the calibration set is sampled deterministically across four length
bins. `reason` and `labels` are not included in model inputs; keep the original
JSONL as a separate accuracy set. The source must normally contain one common
instruction. If production uses a different instruction, set it explicitly:

```bash
PRODUCTION_INSTRUCTION='你的线上 reranker instruction' \
bash scripts/quantize_qwen3_reranker_w8a8_static_310p.sh
```

The safe first export leaves `lm_head` and every `down_proj` in floating point.
After that passes accuracy and runtime checks, `QUANTIZE_DOWN_PROJ=1` creates a
more aggressive candidate in a new `QUANT_MODEL_PATH`. Existing non-empty
model output directories are never overwritten.

To export and immediately run the matching FP16/W8A8 pooling A/B on the first
execution, enable the optional benchmark stage:

```bash
TRAIN_JSONL=/home/reranker_experiment/data/split/train.jsonl \
FLOAT_MODEL_PATH=/home/reranker_experiment/model/Qwen3-Reranker-0.6B \
QUANT_MODEL_PATH=/home/reranker_experiment/model/Qwen3-Reranker-0.6B-W8A8-static-safe \
BENCHMARK_DATA_PATH=/home/reranker_experiment/data/latency_delay \
BENCHMARK_DATASET=0428caption \
BENCHMARK_BACKENDS=pooling \
BENCHMARK_BATCH_SIZES='1 8 16 32' \
RUN_BENCHMARK=1 \
PULL_IMAGE=0 \
bash scripts/quantize_qwen3_reranker_w8a8_static_310p.sh
```

If ModelSlim is already installed, set `INSTALL_MODELSLIM=0` and point
`MODELSLIM_DIR` and `MODELSLIM_VENV` at the pinned checkout/environment. The
generated calibration manifest records input/output SHA-256 values, selected
source indexes, token-length percentiles, and truncation counts.

ModelSlim installation defaults to the Tsinghua PyPI mirror for both
`install.sh` and the pinned Transformers dependencies. Override it when needed:

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
bash scripts/quantize_qwen3_reranker_w8a8_static_310p.sh
```

If the host has no CANN toolkit, run the complete CPU export in the pinned
legacy vLLM-Ascend image instead. The wrapper defaults to the NJU mirror
`quay.nju.edu.cn/ascend/vllm-ascend:v0.9.0rc2`, verifies that the image really
contains the old CANN ModelSlim overlay, and persists the result, calibration
data, checkout, and image-specific venv through host mounts. It does not expose
an NPU or modify the host driver:

```bash
cd /home/reranker_experiment/mem0-vllm
bash scripts/quantize_qwen3_reranker_w8a8_static_310p_container.sh
```

The defaults match `/home/reranker_experiment/data/split/train.jsonl` and the
merged model at
`/home/reranker_experiment/model/qwen3_reranker_06b_lora_merged`. Override
paths normally when needed:

```bash
TRAIN_JSONL=/data/train.jsonl \
FLOAT_MODEL_PATH=/models/qwen3-reranker-merged \
QUANT_MODEL_PATH=/models/qwen3-reranker-w8a8 \
bash scripts/quantize_qwen3_reranker_w8a8_static_310p_container.sh
```

This pinned ModelSlim tag assembles part of its Python package from the CANN
toolkit during `install.sh`. CPU calibration means model tensors stay on the
CPU; it does not remove that install-time package dependency. The workflow
automatically sources `/usr/local/Ascend/ascend-toolkit/set_env.sh` (including
the common `latest` variant) when `ASCEND_HOME_PATH` is not ready. For a custom
CANN installation, pass its environment script explicitly:

```bash
CANN_SET_ENV=/opt/Ascend/ascend-toolkit/set_env.sh \
REINSTALL_MODELSLIM=1 \
bash scripts/quantize_qwen3_reranker_w8a8_static_310p.sh
```

If an earlier run printed `collect packages from CANN installation path:
/python/site-packages/msmodelslim/`, it installed an incomplete wheel because
`ASCEND_HOME_PATH` was empty. Pull this fix and rerun with
`REINSTALL_MODELSLIM=1`; the workflow now checks the exact anti-outlier and PTQ
imports before accepting or stamping an installation. `torch_npu is not
available` remains an expected warning during the CPU export path.

```bash
python scripts/check_qwen3_reranker_w8a8_310p.py \
  --model-path /models/Qwen3-Reranker-0.6B-W8A8
```

The patch keeps the synthetic Qwen3 pooling `score` head unquantized and fixes
the missing `lm_head` prefix in vLLM 0.10.2. These avoid the two empty/missing
quantization-description key failures on this old stack. Start the quantized
A/B run with:

```bash
HOST_MODEL_PATH=/home/reranker_experiment/model/Qwen3-Reranker-0.6B-W8A8 \
HOST_OUTPUT_BASE=/home/reranker_experiment/output/qwen3_310p_w8a8_ab \
MODEL_LABEL=w8a8 \
VLLM_QUANTIZATION=ascend \
BACKENDS="generate pooling" \
PULL_IMAGE=0 \
bash scripts/benchmark_qwen3_reranker_310p.sh
```

Treat W8A8 as successful only when it improves warm p50/p95 or throughput on
the actual production length/batch distribution while preserving the business
ranking metrics. A successful model load alone is not a performance result.

### Atlas 300I / 310P MindIE

MindIE is an alternative 310P backend when vLLM eager-generation latency is
not acceptable. The target stack is:

- host HDK/driver: `24.1.RC2.x` (leave it installed on the host)
- container userspace: MindIE `2.1.RC1`, CANN `8.2.RC1`, ATB Models `2.1.RC1`
- image: `swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.1.RC1-300I-Duo-py311-openeuler24.03-lts`

The service and client stay on `127.0.0.1`; this is local NPU inference, not an
external model API. Pull and start the service from the repository root:

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.1.RC1-300I-Duo-py311-openeuler24.03-lts

ASCEND_RT_VISIBLE_DEVICES=0 \
HOST_MODEL_PATH=/home/reranker_experiment/Qwen3-Reranker-4B \
MAX_LENGTH=8192 \
MAX_BATCH_SIZE=32 \
MAX_PREFILL_TOKENS=32768 \
PULL_IMAGE=0 \
bash scripts/run_mindie_310p_container.sh
```

The launcher backs up the model's original `config.json` as
`config.json.memranker.bak` and changes its dtype to `float16`, which is needed
on 310P. It also configures the ATB backend, one generated token, local-only
HTTP, and disables Prefix Cache because MindIE logprobs cannot be combined with
that feature.

Check service health and one reranker response:

```bash
curl -fsS http://127.0.0.1:1025/health

python - <<'PY'
import json
from urllib.request import Request, urlopen

prompt = (
    '<|im_start|>system\nJudge whether the Document meets the requirements '
    'based on the Query and the Instruct provided. Note that the answer can '
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    '<Instruct>: Given a user query, retrieve relevant documents that answer the query.\n\n'
    '<Query>: capital of China\n\n<Document>: Beijing is the capital of China.'
    '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
)
body = json.dumps({
    'model': 'qwen3-reranker-4b', 'prompt': prompt, 'max_tokens': 1,
    'temperature': 0, 'ignore_eos': True, 'logprobs': 5,
}).encode()
req = Request('http://127.0.0.1:1025/v1/completions', body,
              {'Content-Type': 'application/json'})
print(urlopen(req, timeout=600).read().decode())
PY
```

Run one model against one business dataset:

```bash
BACKEND=mindie \
DATA_ROOT=/home/reranker_experiment/data/latency_delay \
DATASET_NAME=0428caption \
MODEL_PATH=/home/reranker_experiment/Qwen3-Reranker-4B \
MINDIE_MODEL_NAME=qwen3-reranker-4b \
OUTPUT_ROOT=/home/output4b/mindie_0428caption \
BATCH_SIZE=32 \
MAX_REQUEST_CHARS=32000 \
bash scripts/eval_business_matrix_ascend_local.sh
```

`BATCH_SIZE` is the number of concurrent loopback requests. MindIE performs the
dynamic NPU batching. Start with `8`, then compare `16` and `32`; report both
`seconds_per_example` and `examples_per_second` from `metrics.json`. The
`REQUEST_MODE=list` option exists for later MindIE versions, but `concurrent`
is the compatible default for `2.1.RC1`.

MindIE lists the Qwen3 causal architecture on 300I Duo, but does not separately
certify `Qwen3-Reranker-4B`. This path therefore uses the model's official
one-token yes/no reranking prompt. If the image rejects the model architecture
during startup, do not patch ATB model code in place; keep the vLLM eager path
or move to a reranker-native serving stack.

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

To compare two LoCoMo runs and find cases where MemReranker beats a Qwen
soft-label run:

```bash
BETTER_EVAL_DIR=outputs/locomo_mem_4blora_vllm_dp \
WORSE_EVAL_DIR=outputs/locomo_qwen_4blora_vlllm_dp \
BETTER_NAME=mem_reranker \
WORSE_NAME=qwen_soft_label \
TEST_FILE=data/locomo/locomo_qwen3_embedding_candidates.jsonl \
OUTPUT_DIR=outputs/locomo_mem_vs_qwen_cases \
MIN_DELTA=0.2 \
WORSE_MAX_NDCG=0.5 \
bash scripts/compare_locomo_runs.sh
```

This writes:

```text
comparison_summary.json
query_comparison.csv/jsonl
better_cases.csv/jsonl
better_cases_report.md
```

`better_cases_report.md` is the easiest file to read. It shows each query,
both models' `NDCG@10` and first positive rank, plus the top documents from
both runs. Increase `MIN_DELTA` for cleaner high-confidence cases, or set
`WORSE_MAX_NDCG=1.1` to disable the filter that requires the Qwen run to be bad.

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

## GTE reranker on Ascend 310P

`Alibaba-NLP/gte-multilingual-reranker-base` is an encoder-only sequence
classification reranker. The evaluator supports two fully local NPU backends
and preserves the Business Evaluation Matrix output format:

- `BACKEND=vllm` uses vLLM's native `GteNewForSequenceClassification` and
  `LLM.score`, bypassing the model repository's PyTorch SDPA implementation.
- `BACKEND=torch_npu` uses the local Transformers model with the explicit 310P
  PromptFlashAttention patch described below.

### vLLM 0.10.2 container

GTE sequence-classification support first appears in vLLM 0.10.2. Use the
matching 310P image; `v0.10.0rc1-310p` is the generally recommended stable
310P image but does not contain the native GTE reranker architecture.

```text
image        quay.io/ascend/vllm-ascend:v0.10.2rc1-310p
China mirror quay.nju.edu.cn/ascend/vllm-ascend:v0.10.2rc1-310p
vLLM         0.10.2
vllm-ascend  0.10.2rc1
CANN         8.2.RC1 container userspace
host driver  24.1.RC2.x
```

The script mounts the host driver into the container, validates the package
versions and vLLM model registry, performs an NPU tensor smoke test, and then
runs one dataset or the complete matrix with in-process `LLM.score`. It does
not start an HTTP service. On Huawei Cloud 310P hosts it uses privileged mode,
mounts the complete host driver/firmware directories, and sets
`ASCEND_RUNTIME_OPTIONS=NODRV`; mounting only `driver/lib64` can leave HAL
unable to identify the device (`drvErr=87`). If the inference image does not
contain `openpyxl`, the launcher installs `openpyxl==3.1.5` from the Huawei
Cloud PyPI mirror before reading the business Excel files.

The `0.10.2rc1` Ascend plugin has three 310P pooling defects. Its ATB warm-up
calls an unsupported FP32 operation, its custom scheduler mistakes the
encoder-only model's one-block placeholder for a 128-token KV cache, and its
model runner omits encoder-only attention metadata. The launcher applies a
version-checked source patch that skips the optional ATB warm-up, keeps pooling
models on vLLM's native V1 scheduler, and backports the encoder-only metadata
and `_npu_flash_attention` dispatch used by the later 310P implementation.
Generative models continue to use the Ascend scheduler. Set
`PATCH_ATB_WARMUP=0` only for diagnosis or when using a rebuilt image that
already includes all three fixes.

```bash
HOST_REPO_PATH=/home/reranker_experiment/mem0-vllm \
HOST_DATA_PATH=/home/reranker_experiment/data/latency_delay \
HOST_MODEL_PATH=/home/reranker_experiment/model/GTE/GTE \
HOST_OUTPUT_PATH=/home/reranker_experiment/output4b/business_matrix_GTE_vllm \
DATASET=all \
MAX_LENGTH=8192 \
BATCH_SIZE=16 \
PULL_IMAGE=1 \
bash scripts/run_gte_vllm_310p_container.sh
```

For a restricted cloud that already has the image, set `PULL_IMAGE=0`. To use
DaoCloud instead of the Nanjing University mirror, set:

```bash
IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:v0.10.2rc1-310p \
PULL_IMAGE=1 bash scripts/run_gte_vllm_310p_container.sh
```

Inside an already-running `v0.10.2rc1-310p` container, run the same backend
directly:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
BACKEND=vllm \
DATA_ROOT=/workspace/data/latency_delay \
MODEL_PATH=/workspace/model/GTE/GTE \
OUTPUT_ROOT=/workspace/output4b/business_matrix_GTE_vllm \
DATASET=all DTYPE=float16 \
BATCH_SIZE=16 MAX_LENGTH=8192 \
MAX_NUM_BATCHED_TOKENS=8192 MAX_NUM_SEQS=16 \
ENFORCE_EAGER=1 \
bash scripts/eval_business_gte_310p.sh
```

Keep `ENFORCE_EAGER=1` for the first 310P run. This disables vLLM graph
capture, but still uses vLLM's native GTE model instead of Hugging Face SDPA.
The evaluator passes `truncate_prompt_tokens=MAX_LENGTH`, so long recall
documents are truncated rather than rejected by vLLM.

After patching, the engine log must not say that
`vllm_ascend.core.scheduler.AscendScheduler` was selected. Warnings that an
ordinary 171-216 token input exceeds a limit of 128 indicate that the old
container process is still running or the launcher patch was disabled.

### torch-npu PFA fallback

The default `ATTENTION_BACKEND=pfa` path explicitly replaces the model's
PyTorch SDPA call with `torch_npu.npu_prompt_flash_attention`. This is required
because automatic PyTorch SDPA dispatch does not select fused attention on
310P. The evaluator converts GTE's additive padding mask to the full BOOL mask
required by the 310P operator and reuses it across all encoder layers.

The conservative Python 3.9 compatibility line used by this repository is:

```text
Ascend host driver  24.1.RC2.x
CANN                8.0.RC2 toolkit + 310P kernels
Python              3.9.x
torch               2.1.0
torch-npu           2.1.0.post6
transformers        4.39.1
tokenizers          0.15.1
```

Do not install the CANN NNAL package for this backend. NNAL is used by ATB and
MindIE components; this evaluator only needs the CANN toolkit and the matching
310P binary kernels. Do not install `vllm`, `vllm-ascend`, `triton`, `xformers`,
or any `nvidia-*` wheels into the GTE environment.

Install the CANN user-space packages matching the host architecture. Keep the
cloud host driver unchanged:

```bash
chmod +x Ascend-cann-toolkit_8.0.RC2_linux-aarch64.run
chmod +x Ascend-cann-kernels-310p_8.0.RC2_linux-aarch64.run
./Ascend-cann-toolkit_8.0.RC2_linux-aarch64.run --install
./Ascend-cann-kernels-310p_8.0.RC2_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

Use the `x86_64` packages instead when `uname -m` returns `x86_64`. Create a
clean Python 3.9 environment and install the matching cp39 torch wheels before
the remaining packages:

```bash
python3.9 -m venv /home/reranker_experiment/gte310env
source /home/reranker_experiment/gte310env/bin/activate
python -m pip install --upgrade "pip<26" setuptools==68.2.2 wheel==0.41.3

pip install torch==2.1.0 torch-npu==2.1.0.post6 \
  -i https://mirrors.huaweicloud.com/repository/pypi/simple \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi

pip install -r requirements-ascend-gte-310p-py39.txt \
  -i https://mirrors.huaweicloud.com/repository/pypi/simple \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

If pip tries to replace the already installed torch wheels, install the
non-framework dependencies separately with the versions in
`requirements-ascend-gte-310p-py39.txt`; torch and torch-npu must remain an
exact pair.

The model directory must contain at least `config.json`, `model.safetensors`,
`tokenizer.json`, and `tokenizer_config.json`. Validate the NPU and run a
semantic model smoke test before the full experiment:

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /home/reranker_experiment/gte310env/bin/activate

ASCEND_RT_VISIBLE_DEVICES=0 python scripts/check_gte_310p_env.py \
  --model_path /home/reranker_experiment/model/gte-multilingual-reranker-base \
  --device npu:0 \
  --attention_backend pfa \
  --jit_compile
```

Run exactly one GTE model on one business dataset:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
DATA_ROOT=/home/reranker_experiment/data/latency_delay \
MODEL_PATH=/home/reranker_experiment/model/gte-multilingual-reranker-base \
DATASET=0428caption \
DTYPE=fp16 ATTENTION_BACKEND=pfa JIT_COMPILE=1 \
BATCH_SIZE=16 MAX_LENGTH=512 \
bash scripts/eval_business_gte_310p.sh
```

Use `ATTENTION_BACKEND=eager JIT_COMPILE=0` as the compatibility and accuracy
baseline. Do not use `ATTENTION_BACKEND=sdpa` on 310P; it takes the unsupported
math/format-conversion path that reports `can not cast format when output is
input`. PFA on 310P requires `DTYPE=fp16`.

Valid dataset names are `0428caption`, `0428keyword`, and `0625caption`. Start
with FP16 on 310P. After the 512-token run passes, increase `MAX_LENGTH` to
1024, 2048, or 8192 and reduce `BATCH_SIZE` when necessary.

Set `DATASET=all` to run the same GTE model over all three business datasets
and generate `summary_metrics.csv`, `summary_metrics.json`, and
`summary_metrics.xlsx` under one output root:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
DATA_ROOT=/home/reranker_experiment/data/latency_delay \
MODEL_PATH=/home/reranker_experiment/model/gte-multilingual-reranker-base \
DATASET=all DTYPE=fp16 ATTENTION_BACKEND=pfa JIT_COMPILE=1 \
BATCH_SIZE=16 MAX_LENGTH=512 \
bash scripts/eval_business_gte_310p.sh
```
