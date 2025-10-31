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


# --- Install Specific Versions ---
# Note: Ubuntu 22.04 has GCC 12 in default repositories, no PPA needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python3-pip \
        g++-${GCC_VERSION} \
        gcc-${GCC_VERSION} && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} 100 && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Expose JupyterLab port
EXPOSE 8888

# Start JupyterLab by default
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.disable_check_xsrf=True", "--ServerApp.open_browser=False"]