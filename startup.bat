@echo off
setlocal enabledelayedexpansion

REM Resolve the current directory
set CURRENT_DIR=%~dp0
if "%CURRENT_DIR:~-1%"=="\" set CURRENT_DIR=%CURRENT_DIR:~0,-1%

REM Named volumes
set CLICKHOUSE_DATA_VOLUME=tictactoe_clickhouse_data
set CLICKHOUSE_LOGS_VOLUME=tictactoe_clickhouse_logs

REM Ensure named volumes exist
docker volume inspect %CLICKHOUSE_DATA_VOLUME% >nul 2>&1
if errorlevel 1 docker volume create %CLICKHOUSE_DATA_VOLUME%

docker volume inspect %CLICKHOUSE_LOGS_VOLUME% >nul 2>&1
if errorlevel 1 docker volume create %CLICKHOUSE_LOGS_VOLUME%

REM Detect GPU
set GPU_ARGS=
where nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    echo Detected NVIDIA GPU → enabling CUDA
    set GPU_ARGS=--gpus all
) else (
    echo No NVIDIA GPU detected → running on CPU
)

REM Run container
docker run -it --rm ^
    %GPU_ARGS% ^
    --name ttt_ai ^
    -p 8888:8888 ^
    -p 9000:9000 ^
    -p 8123:8123 ^
    -p 3030:3030 ^
    -p 5050:5050 ^
    -v "%CURRENT_DIR%:/app" ^
    -v %CLICKHOUSE_DATA_VOLUME%:/var/lib/clickhouse ^
    -v %CLICKHOUSE_LOGS_VOLUME%:/var/log/clickhouse-server ^
    -e JUPYTER_ENABLE_LAB=yes ^
    tictactoe.ai

endlocal
