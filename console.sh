#!/bin/bash
# Move to the parent directory of where this script is located
cd "$(dirname "$0")/.." || exit 1

# Execute into the running container
docker exec -it ttt_ai /bin/bash
