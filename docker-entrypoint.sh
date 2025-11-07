#!/bin/bash
set -e

# Fix permissions for ClickHouse directories
chown -R clickhouse:clickhouse /var/lib/clickhouse /var/log/clickhouse-server 2>/dev/null || true

# Start ClickHouse server in the background as clickhouse user
echo "Starting ClickHouse server..."
su -s /bin/bash clickhouse -c "clickhouse-server --config-file=/etc/clickhouse-server/config.xml" &
CLICKHOUSE_PID=$!

# Wait for ClickHouse to be ready
echo "Waiting for ClickHouse to initialize..."
for i in {1..30}; do
    # Check if ClickHouse process is still running
    if ! kill -0 $CLICKHOUSE_PID 2>/dev/null; then
        echo "ERROR: ClickHouse process died during startup!"
        echo "Check logs at /var/log/clickhouse-server/clickhouse-server.err.log"
        tail -n 50 /var/log/clickhouse-server/clickhouse-server.err.log 2>/dev/null || true
        exit 1
    fi
    
    # Check if ClickHouse is ready to accept connections
    if clickhouse-client --query "SELECT 1" > /dev/null 2>&1; then
        echo "ClickHouse is ready!"
        break
    fi
    
    if [ $i -eq 30 ]; then
        echo "Warning: ClickHouse may not be fully ready"
    fi
    sleep 1
done

# Start JupyterLab
echo "Starting JupyterLab..."
exec jupyter lab --ip=0.0.0.0 --port=8888 --allow-root \
    --ServerApp.token='' \
    --ServerApp.password='' \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.open_browser=False

