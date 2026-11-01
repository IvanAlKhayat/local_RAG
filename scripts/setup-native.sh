#!/usr/bin/env bash
# One-time native installation for macOS/Linux with Python 3.12.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "Python 3.12 is required. On a Mac: brew install python@3.12" >&2
  exit 1
fi

if [[ ! -f config.yaml ]]; then
  cp config.example.yaml config.yaml
  echo "Created config.yaml; edit source paths or vLLM endpoints as needed."
fi

mkdir -p documents data

if [[ ! -x .venv-rag312/bin/python ]]; then
  python3.12 -m venv .venv-rag312
fi

echo "Installing the LocalRAG backend..."
.venv-rag312/bin/python -m pip install -e .

if [[ ! -x .venv-webui312/bin/python ]]; then
  python3.12 -m venv .venv-webui312
fi

echo "Installing Open WebUI in a separate Python environment..."
.venv-webui312/bin/python -m pip install open-webui

echo
echo "Installation complete. Start the Ollama application, then run:"
echo "  bash scripts/run-native.sh"
