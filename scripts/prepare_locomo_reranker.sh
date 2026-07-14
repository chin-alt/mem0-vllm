#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="${INPUT_FILE:-data/locomo/locomo10.json}"
OUTPUT_FILE="${OUTPUT_FILE:-data/locomo/locomo_qwen3_embedding_candidates.jsonl}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-100}"
RETRIEVAL_BACKEND="${RETRIEVAL_BACKEND:-embedding}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_QUERIES="${MAX_QUERIES:-0}"
MAX_QA_PER_SAMPLE="${MAX_QA_PER_SAMPLE:-0}"
DOC_MAX_CHARS="${DOC_MAX_CHARS:-0}"
INDEX_DOC_MAX_CHARS="${INDEX_DOC_MAX_CHARS:-2048}"
EMBEDDING_MODEL_NAME_OR_PATH="${EMBEDDING_MODEL_NAME_OR_PATH:-Qwen/Qwen3-Embedding-0.6B}"
EMBEDDING_QUERY_INSTRUCTION="${EMBEDDING_QUERY_INSTRUCTION:-Given a long-term conversation memory question, retrieve relevant dialogue turns.}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-32}"
EMBEDDING_SEARCH_BATCH_SIZE="${EMBEDDING_SEARCH_BATCH_SIZE:-64}"
EMBEDDING_MAX_LENGTH="${EMBEDDING_MAX_LENGTH:-512}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-auto}"
EMBEDDING_MULTI_PROCESS="${EMBEDDING_MULTI_PROCESS:-0}"
EMBEDDING_DEVICES="${EMBEDDING_DEVICES:-}"
EMBEDDING_CHUNK_SIZE="${EMBEDDING_CHUNK_SIZE:-0}"
EMBEDDING_SEARCH_DEVICE="${EMBEDDING_SEARCH_DEVICE:-auto}"
EMBEDDING_SEARCH_DTYPE="${EMBEDDING_SEARCH_DTYPE:-auto}"
EMBEDDING_DTYPE="${EMBEDDING_DTYPE:-auto}"
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-data/locomo/embedding_cache}"
EMBEDDING_LOCAL_FILES_ONLY="${EMBEDDING_LOCAL_FILES_ONLY:-0}"
ENSURE_POSITIVES="${ENSURE_POSITIVES:-0}"
INCLUDE_NO_EVIDENCE="${INCLUDE_NO_EVIDENCE:-0}"
INCLUDE_ANSWER_METADATA="${INCLUDE_ANSWER_METADATA:-0}"
INSTRUCTION="${INSTRUCTION:-Given a long-term conversation memory question, retrieve dialogue turns that contain evidence needed to answer the question.}"
PYTHON_BIN="${PYTHON_BIN:-python}"

EXTRA_ARGS=()
if [[ "${ENSURE_POSITIVES}" == "1" ]]; then
  EXTRA_ARGS+=(--ensure_positives)
fi
if [[ "${INCLUDE_NO_EVIDENCE}" == "1" ]]; then
  EXTRA_ARGS+=(--include_no_evidence)
fi
if [[ "${INCLUDE_ANSWER_METADATA}" == "1" ]]; then
  EXTRA_ARGS+=(--include_answer_metadata)
fi
if [[ "${EMBEDDING_LOCAL_FILES_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--embedding_local_files_only)
fi
if [[ "${EMBEDDING_MULTI_PROCESS}" == "1" ]]; then
  EXTRA_ARGS+=(--embedding_multi_process)
fi

"${PYTHON_BIN}" src/prepare_locomo_reranker.py \
  --input_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --retrieval_backend "${RETRIEVAL_BACKEND}" \
  --candidate_top_k "${CANDIDATE_TOP_K}" \
  --max_samples "${MAX_SAMPLES}" \
  --max_queries "${MAX_QUERIES}" \
  --max_qa_per_sample "${MAX_QA_PER_SAMPLE}" \
  --doc_max_chars "${DOC_MAX_CHARS}" \
  --index_doc_max_chars "${INDEX_DOC_MAX_CHARS}" \
  --embedding_model_name_or_path "${EMBEDDING_MODEL_NAME_OR_PATH}" \
  --embedding_query_instruction "${EMBEDDING_QUERY_INSTRUCTION}" \
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
  --embedding_search_batch_size "${EMBEDDING_SEARCH_BATCH_SIZE}" \
  --embedding_max_length "${EMBEDDING_MAX_LENGTH}" \
  --embedding_device "${EMBEDDING_DEVICE}" \
  --embedding_devices "${EMBEDDING_DEVICES}" \
  --embedding_chunk_size "${EMBEDDING_CHUNK_SIZE}" \
  --embedding_search_device "${EMBEDDING_SEARCH_DEVICE}" \
  --embedding_search_dtype "${EMBEDDING_SEARCH_DTYPE}" \
  --embedding_dtype "${EMBEDDING_DTYPE}" \
  --embedding_cache_dir "${EMBEDDING_CACHE_DIR}" \
  --instruction "${INSTRUCTION}" \
  "${EXTRA_ARGS[@]}"
