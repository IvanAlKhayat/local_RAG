#!/usr/bin/env bash
# Run the backend first, wait for its health check, then launch Open WebUI.
# The script owns both processes and stops them together on Ctrl+C.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ ! -x .venv-rag312/bin/python || ! -x .venv-webui312/bin/open-webui ]]; then
  echo "Install first: bash scripts/setup-native.sh" >&2
  exit 1
fi
if [[ ! -f config.yaml ]]; then
  echo "Missing config.yaml; run bash scripts/setup-native.sh" >&2
  exit 1
fi
if ! curl -fsS --max-time 3 http://127.0.0.1:11434/api/tags >/dev/null; then
  echo "Ollama is not running. Launch Ollama on your Mac first." >&2
  exit 1
fi

# Avoid running two servers on one port with different configurations.
if curl -fsS --max-time 2 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
  echo "Port 8000 already serves a RAG API. Stop that instance first." >&2
  exit 1
fi
if curl -fsS --max-time 2 http://127.0.0.1:8080/ >/dev/null 2>&1; then
  echo "Port 8080 already serves a web UI. Stop that instance first." >&2
  exit 1
fi

rag_pid=""
webui_pid=""
cleanup() {
  [[ -z "$webui_pid" ]] || kill "$webui_pid" 2>/dev/null || true
  [[ -z "$rag_pid" ]] || kill "$rag_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting LocalRAG on http://127.0.0.1:8000 ..."
LOCALRAG_CONFIG="$project_dir/config.yaml" \
  .venv-rag312/bin/python -m localrag.cli serve &
rag_pid=$!

# Web UI must not start before its model-discovery endpoint is ready.
ready=false
for ((attempt=0; attempt<30; attempt++)); do
  if curl -fsS --max-time 2 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    ready=true
    break
  fi
  if ! kill -0 "$rag_pid" 2>/dev/null; then
    echo "LocalRAG exited during startup. Check the error above." >&2
    exit 1
  fi
  sleep 1
done
if [[ "$ready" != true ]]; then
  echo "LocalRAG did not become ready within 30 seconds." >&2
  exit 1
fi

echo "Starting Open WebUI on http://127.0.0.1:8080 ..."
# Explicit environment values avoid stale provider settings from old installs.
# Open WebUI's optional document embeddings use Ollama, not PyTorch/MPS.
# Chat still routes through LocalRAG's OpenAI-compatible API.
ENABLE_PERSISTENT_CONFIG=false \
ENABLE_OPENAI_API=true \
ENABLE_OLLAMA_API=false \
OPENAI_API_BASE_URLS=http://127.0.0.1:8000/v1 \
OPENAI_API_KEYS=localrag-local \
OPENAI_API_CONFIGS='{"0":{"enable":true,"model_ids":[]}}' \
RAG_EMBEDDING_ENGINE=ollama \
RAG_EMBEDDING_MODEL=embeddinggemma \
OLLAMA_BASE_URL=http://127.0.0.1:11434 \
DATA_DIR="$project_dir/data/open-webui" \
  .venv-webui312/bin/open-webui serve --port 8080 &
webui_pid=$!

echo
echo "Chat:      http://127.0.0.1:8080"
echo "Documents: http://127.0.0.1:8000/manage"
echo "Models:    http://127.0.0.1:8000/v1/models"
echo "Press Ctrl+C to stop both services."

while kill -0 "$rag_pid" 2>/dev/null && kill -0 "$webui_pid" 2>/dev/null; do
  sleep 2
done
echo "A service exited; stopping the other one." >&2
exit 1
