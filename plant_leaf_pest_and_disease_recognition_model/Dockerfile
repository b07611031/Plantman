# 使用 NVIDIA 的 PyTorch 基礎映像，已包含 CUDA 和 PyTorch
FROM nvcr.io/nvidia/pytorch:21.08-py3

# 設定工作目錄
WORKDIR /app

# 複製所有需要的檔案到容器中
COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py
COPY models /app/models
COPY utils /app/utils
COPY data /app/data

# 更新並安裝必要的系統依賴
RUN apt-get update -y && apt-get install -y \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    lsof && \
    rm -rf /var/lib/apt/lists/*  # 清理 apt 緩存以減少映像大小

# 安裝 Python 套件，並確保使用最新版本的 pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 設置環境變數
ENV PYTHONIOENCODING=utf-8

# 開放服務埠（假設服務在 5000 埠上運行）
EXPOSE 5000

# 設置容器啟動時的指令，這裡啟動 YOLOv9 plant symptoms detection service
CMD ["python", "app.py"]