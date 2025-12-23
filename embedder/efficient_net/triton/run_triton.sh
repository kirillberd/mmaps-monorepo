
set -euo pipefail

MODEL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model-repository"

docker run --rm \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v "${MODEL_REPO}:/models" \
  nvcr.io/nvidia/tritonserver:25.02-py3 \
  tritonserver \
    --model-repository=/models \
    --log-verbose=0 \
    --strict-model-config=true
