FROM ubuntu:22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.11 \
    GCC_VERSION=12 \
    CMAKE_VERSION=3.25.1 \
    PATH="/opt/venv/bin:$PATH"

# --- Install Base Tools & GPG (Fixes PPA Import Errors) ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        ca-certificates \
        curl wget git gnupg lsb-release && \
    rm -rf /var/lib/apt/lists/*

# Add Deadsnakes PPA for Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gnupg dirmngr ca-certificates curl && \
    mkdir -p /etc/apt/keyrings /root/.gnupg && \
    chmod 700 /root/.gnupg && \
    gpg --no-default-keyring --keyring /etc/apt/keyrings/deadsnakes.gpg \
        --keyserver keyserver.ubuntu.com \
        --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 && \
    echo "deb [signed-by=/etc/apt/keyrings/deadsnakes.gpg] http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" \
        > /etc/apt/sources.list.d/deadsnakes.list && \
    apt-get update

# Add ClickHouse Repository
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-transport-https && \
    curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' | gpg --dearmor -o /etc/apt/keyrings/clickhouse-keyring.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/clickhouse-keyring.gpg] https://packages.clickhouse.com/deb stable main" \
        > /etc/apt/sources.list.d/clickhouse.list && \
    apt-get update


# --- Install Specific Versions ---
# Note: Ubuntu 22.04 has GCC 12 in default repositories, no PPA needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python3-pip \
        g++-${GCC_VERSION} \
        gcc-${GCC_VERSION} \
        clickhouse-server \
        clickhouse-client && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} 100 && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# --- Install Node.js and PM2 ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    npm install -g pm2 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --ignore-installed -r requirements.txt
    #export PYTHONPATH=/app/DRL:$PYTHONPATH
    #python -m pip install --upgrade --ignore-installed flask

# Copy application code (Statistic folder for API server, Demo folder for frontend)
COPY Statistic/ ./Statistic/
COPY Demo/ ./Demo/

# Copy entrypoint script and fix line endings (Windows CRLF -> Unix LF)
COPY docker-entrypoint.sh /usr/local/bin/
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy ClickHouse configuration files
COPY clickhouse-config.xml /etc/clickhouse-server/config.d/
COPY clickhouse-users.xml /etc/clickhouse-server/users.d/

# Create ClickHouse data directory and set permissions
RUN mkdir -p /var/lib/clickhouse /var/log/clickhouse-server /etc/clickhouse-server/users.d /etc/clickhouse-server/config.d && \
    chown -R clickhouse:clickhouse /var/lib/clickhouse /var/log/clickhouse-server

# Define volumes for data persistence
VOLUME ["/var/lib/clickhouse", "/var/log/clickhouse-server"]

# Expose JupyterLab, ClickHouse, Node.js server, and Python API server ports
EXPOSE 8888 9000 8123 3030 5050

# Start both ClickHouse and JupyterLab
CMD ["/usr/local/bin/docker-entrypoint.sh"]