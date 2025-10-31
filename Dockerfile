FROM debian:bookworm-slim

# Specify versions
ENV PYTHON_VERSION=3.11 \
    GCC_VERSION=12 \
    CMAKE_VERSION=3.25.1

# Install specific versions of Python, pip, and C++ build tools
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python3-pip \
        python3-venv \
        g++-${GCC_VERSION} \
        build-essential \
        cmake=${CMAKE_VERSION}* \
        git \
        curl && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} 100 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 100

# Add useful aliases
RUN echo "alias python='python3'" >> /root/.bashrc && \
    echo "alias pip='pip3'" >> /root/.bashrc && \
    echo "alias c++='g++ -std=c++17 -O3 -pthread -o a.out'" >> /root/.bashrc && \
    echo "alias cxxrun='f(){ g++ -std=c++17 -O3 -pthread -o a.out \"$1\" && ./a.out; }; f'" >> /root/.bashrc

# Create working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Precompile your C++ generator
COPY Dataset/state_generation.cpp /tmp/state_generation.cpp
RUN g++ -std=c++17 -O3 -pthread -o /usr/local/bin/state_generation /tmp/state_generation.cpp && \
    rm /tmp/state_generation.cpp

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["python3", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
