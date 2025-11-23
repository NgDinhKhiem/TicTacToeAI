#!/bin/bash
set -e

# Resolve script directory to allow relative volume paths
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Named volumes for persistent ClickHouse data & logs
CLICKHOUSE_DATA_VOLUME="tictactoe_clickhouse_data"
CLICKHOUSE_LOGS_VOLUME="tictactoe_clickhouse_logs"

# Ensure named volumes exist
docker volume inspect $CLICKHOUSE_DATA_VOLUME >/dev/null 2>&1 || docker volume create $CLICKHOUSE_DATA_VOLUME
docker volume inspect $CLICKHOUSE_LOGS_VOLUME >/dev/null 2>&1 || docker volume create $CLICKHOUSE_LOGS_VOLUME

# Detect if NVIDIA GPU is available (Linux only)
GPU_ARGS=""
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Detected NVIDIA GPU → enabling CUDA"
    GPU_ARGS="--gpus all"
else
    echo "No NVIDIA GPU detected → running on CPU"
fi

# Run container
docker run -it --rm \
    $GPU_ARGS \
    --name ttt_ai \
    -p 8888:8888 \
    -p 9000:9000 \
    -p 8123:8123 \
    -p 3030:3030 \
    -p 5050:5050 \
    -v "$CURRENT_DIR:/app" \
    -v $CLICKHOUSE_DATA_VOLUME:/var/lib/clickhouse \
    -v $CLICKHOUSE_LOGS_VOLUME:/var/log/clickhouse-server \
    -e JUPYTER_ENABLE_LAB=yes \
    tictactoe.ai
