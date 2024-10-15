# RAG + LLM

## 安裝要求
請確保已安裝以下內容：

- Python 3.11+
- Pytorch 2.4.0+（支援GPU）
- CUDA 12.1+
- torch 2.4.0+cu121
- ollama 0.2.6
## RAG 設置：

### 環境準備：

1. 下載 Open WebUI/pipelines 的 repository：
   ```bash
   git clone https://github.com/open-webui/pipelines.git
   ```

2. 使用 `nvidia/cuda:12.1.0-devel-ubuntu22.04` 映像，安裝 Python 3.11.9 並打包映像：
   ```bash
   docker run -it --name <pipeline_base_image> nvidia/cuda:12.1.0-devel-ubuntu22.04 /bin/bash 
   # 在容器內安裝 Python 3.11.9
   docker commit <pipeline_base_image>
   ```

3. 修改 Open WebUI/pipeline 的 Dockerfile 基礎映像，替換為剛剛打包好的映像，然後重新編譯：
   ```bash
   # 修改 Dockerfile 中的基礎映像
   # FROM python:3.11-slim-bookworm as base 改為 FROM <pipeline_base_image> as base

   docker build -t <pipeline_image> --build-arg  USE_CUDA=true --build-arg USE_CUDA_VER=cu121 .
   docker run -d -p 9099:9099 <pipeline_image>
   ```

### 文件預處理：
- 使用 Embedding 模型編碼所有文本文件，並將文件向量儲存下來。

### 流程：
1. 輸入：使用者查詢（query）
2. 使用 Embedding 模型對使用者查詢進行編碼。
3. 計算使用者查詢向量與文件向量的餘弦相似度，選擇最相似的 top_k 文件。
4. 將文件文本與使用者查詢包裝成一個列表：`[(document, query) for document in documents]`。
5. 使用 reranker 模型從列表中選出最終的 top_1 文件。

## LLM 設置：

### 預處理：
1. 使用 llama.cpp 將 LLM 的 PyTorch 模型轉換為 `.gguf` 格式，且不進行量化。
2. 使用 `ollama` Docker 映像（ollama/ollama）來設置 ollama 的運行環境：
   ```bash
   docker run -d -v ollama:/root/.ollama -p 11434:11434 ollama/ollama
   ollama create -f Breeze_7B_Instruct_v1_0 -f ModFile
   ```

### 流程：
1. 使用模板將 prompt 輸入模型：
   ```python
   prompt = (
     f"請根據以下資訊回答使用者的問題。如果問題與資訊無關，請忽略資訊並直接回答使用者問題。\n"
     f"資訊：\n```\n{document}\n```\n"
     f"使用者問題：\n```\n{user_query}\n```\n"
   )
   ```
2. 將模型的輸出回傳給使用者。

## 模型選擇
- Embedding 模型：`BAAI/bge-large-zh-v1.5`
- Reranker 模型：`BAAI/bge-reranker-v2-m3`
- LLM 模型：`MediaTek-Research/Breeze-7B-Instruct-v1_0`

## 目前嘗試與速度測試
- 嘗試將模型轉換為 ONNX 或 TensorRT 來加速運行。
- 速度測試結果：
  - RAG：對 512 個段落進行查詢，平均每次查詢耗時 6.34 秒。
  - LLM：待測試。

## API 範例
```python
headers = {'Content-Type': 'application/json'}
payload = {'user_message': user_message}
response = requests.post('http://pipelines.yfshih.toolmenlab.bime.ntu.edu.tw/rag', headers=headers, json=payload)
```
