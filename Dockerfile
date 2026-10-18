FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive MPLCONFIGDIR=/tmp/matplotlib CUDA_HOME=/usr/local/cuda
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git ninja-build ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /eval
COPY requirements-tested.txt /eval/
RUN python3 -m pip install --no-cache-dir -r requirements-tested.txt
COPY . /eval
RUN python3 -m pip install --no-deps .
ENTRYPOINT ["gs-agent-eval"]
