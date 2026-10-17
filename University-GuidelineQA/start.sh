#!/usr/bin/env sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if ! command -v docker >/dev/null 2>&1; then
  echo 'Install Docker with the Compose plugin, then run sh start.sh again.' >&2
  exit 1
fi
if [ -f runtime/docker-images-amd64.tar ] && [ "$(uname -m)" = "x86_64" ]; then
  if ! docker image inspect ghcr.io/ggml-org/llama.cpp:server-b11429 >/dev/null 2>&1; then
    docker load --input runtime/docker-images-amd64.tar
  fi
fi
docker compose up -d --wait --wait-timeout 180
echo "Guideline Q&A is ready: http://localhost:${PORT:-7860}"
