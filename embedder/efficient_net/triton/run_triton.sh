
set -euo pipefail

MODEL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_repository"

docker run --rm \
  -p 8010:8000 \
  -p 8011:8001 \
  -p 8012:8002 \
  -v "${MODEL_REPO}:/models" \
  nvcr.io/nvidia/tritonserver:25.02-py3 \
  tritonserver \
    --model-repository=/models \
    --log-verbose=0 \
    --disable-auto-complete-config
