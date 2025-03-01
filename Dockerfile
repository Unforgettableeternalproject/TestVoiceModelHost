# 1. 使用適合 PyTorch CUDA 12.1 的基礎映像
ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04
FROM ${BASE_IMAGE} as base

# 2. 安裝 Python 3.10 和必要系統依賴（FFMPEG、音訊處理庫）
ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Taipei
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# 3. 設定 Python 連結
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# 4. 設定工作目錄
WORKDIR /app

RUN pip install --default-timeout=100 --upgrade pip==23.3.2

# 5. 複製 `requirements.txt` 並安裝 Python 依賴
COPY requirements.txt /app/

# 6. 安裝 Python 套件（包括根據 OS 自動選擇 `faiss`）
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# 7. 確保 `tts/weights` 權重目錄可讀寫
COPY . /app
RUN chmod -R 777 /app/tts/weights

# 8. 設定環境變數
ENV MODEL_PATH="/app/tts/weights"
ENV SAMPLE_RATE=22050

# 9. 測試 PyTorch 是否可用（確認 CUDA 是否啟動）
RUN python3 -c "import torch; print(f'Using device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"

# 10. 開放 Flask 伺服器的 5000 端口
EXPOSE 5000

# 11. 設定容器啟動指令
CMD ["python3", "/app/Backend/server.py"]
