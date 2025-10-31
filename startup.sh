#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run -it --rm --name ttt_ai \
    -p 8888:8888 \
    -v "$CURRENT_DIR:/app" \
    -w /app \
    tictactoe.ai