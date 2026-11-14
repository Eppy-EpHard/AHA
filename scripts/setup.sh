#!/usr/bin/env bash
# One-time setup, after `uv sync`: the practice problems (from the original paper's logs) into data/practice/, and
# the base models at the revisions every number here was measured on, into models/.
#   bash scripts/setup.sh             # both models
#   bash scripts/setup.sh 0.8b        # one of them
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d third_party/lm_understand-logs ] || \
  git clone --depth 1 https://github.com/RikiyaT/lm_understand-logs.git third_party/lm_understand-logs
mkdir -p data/practice
cp third_party/lm_understand-logs/exp8_math/practice/{gsm8k_train.jsonl,math_train.jsonl} data/practice/

download() {   # name revision
  uv run python -c "from huggingface_hub import snapshot_download as d; d('Qwen/$1', revision='$2', local_dir='models/$1')"
  echo "$2" > "models/$1/.revision"
}
case "${1:-all}" in
  0.8b) download Qwen3.5-0.8B 2fc06364715b967f1860aea9cf38778875588b17 ;;
  2b)   download Qwen3.5-2B 15852e8c16360a2fea060d615a32b45270f8a8fc ;;
  all)  download Qwen3.5-0.8B 2fc06364715b967f1860aea9cf38778875588b17
        download Qwen3.5-2B 15852e8c16360a2fea060d615a32b45270f8a8fc ;;
  *)    echo "usage: bash scripts/setup.sh [0.8b|2b|all]" >&2; exit 1 ;;
esac
