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

# Run container
docker run -it --rm \
    --name ttt_ai \
    -p 8888:8888 \
    -p 9000:9000 \
    -p 8123:8123 \
    -v "$CURRENT_DIR:/app" \
    -v $CLICKHOUSE_DATA_VOLUME:/var/lib/clickhouse \
    -v $CLICKHOUSE_LOGS_VOLUME:/var/log/clickhouse-server \
    -e JUPYTER_ENABLE_LAB=yes \
    tictactoe.ai
