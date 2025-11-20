@echo off
setlocal enabledelayedexpansion

REM Resolve the current directory
set CURRENT_DIR=%~dp0
REM Remove trailing backslash if present
if "%CURRENT_DIR:~-1%"=="\" set CURRENT_DIR=%CURRENT_DIR:~0,-1%

REM Named volumes for persistent ClickHouse data & logs
set CLICKHOUSE_DATA_VOLUME=tictactoe_clickhouse_data
set CLICKHOUSE_LOGS_VOLUME=tictactoe_clickhouse_logs

REM Ensure named volumes exist
docker volume inspect %CLICKHOUSE_DATA_VOLUME% >nul 2>&1
if errorlevel 1 docker volume create %CLICKHOUSE_DATA_VOLUME%

docker volume inspect %CLICKHOUSE_LOGS_VOLUME% >nul 2>&1
if errorlevel 1 docker volume create %CLICKHOUSE_LOGS_VOLUME%

REM Run container
docker run -it --rm ^
    --name ttt_ai ^
    -p 8888:8888 ^
    -p 9000:9000 ^
    -p 8123:8123 ^
    -p 3000:3000 ^
    -p 5000:5000 ^
    -v "%CURRENT_DIR%:/app" ^
    -v %CLICKHOUSE_DATA_VOLUME%:/var/lib/clickhouse ^
    -v %CLICKHOUSE_LOGS_VOLUME%:/var/log/clickhouse-server ^
    -e JUPYTER_ENABLE_LAB=yes ^
    tictactoe.ai

endlocal


