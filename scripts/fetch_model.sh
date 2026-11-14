#!/usr/bin/env bash
# Download a Qwen3.5 base model into models/<name>/ at a pinned revision, and record the revision next to it.
#   bash scripts/fetch_model.sh                          # Qwen3.5-0.8B at the revision every 0.8B number used
#   bash scripts/fetch_model.sh Qwen3.5-2B <revision>    # any other size: the revision is required
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:-Qwen3.5-0.8B}
REV=${2:-}
if [ -z "$REV" ]; then
  [ "$NAME" = "Qwen3.5-0.8B" ] || { echo "give a revision for $NAME (configs/ pins the ones used)" >&2; exit 1; }
  REV=2fc06364715b967f1860aea9cf38778875588b17
fi
python - "$NAME" "$REV" <<'PY'
import sys
from huggingface_hub import snapshot_download
name, rev = sys.argv[1], sys.argv[2]
print(snapshot_download(f"Qwen/{name}", revision=rev, local_dir=f"models/{name}"))
PY
echo "$REV" > "models/$NAME/.revision"
