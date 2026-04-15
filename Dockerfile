FROM nvcr.io/nvidia/tensorrt:24.01-py3

ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir "torch>=2.0.0" "torchvision>=0.15.0" --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install -r requirements.txt --no-cache-dir

COPY . .

CMD ["/bin/bash"]


