# RAG+LLM

## RAG:
預處理：
- 使用Embedding model encodes 所有text documents 並將document vectors存起來。
流程：
- Input: 使用者query
- 使用Embedding model encodes user query
- 將user query vector與document vectors計算cosine simularity, 選出最相似的top_k documents
- 將douments text與user query包成一個list: [(document, query) for document in documents]
- 使用reranker model將list計算出最終要取出的top_1 document.

## LLM:
預處理：
- 使用llama.cpp將LLM pytorch model轉成.gguf格式，不使用quaternization
- 使用ollama image: ollama/ollama 架設ollama docker環境
- ollama運行LLM model, ModFile為模型的基礎參數，這邊使用與model官網相同的設定:
```
ollama create -f Breeze_7B_Instruct_v1_0 -f ModFile
```
流程：
- 使用template 將prompt輸入model:
```
prompt = (
  f"請根據以下資訊回答使用者的問題。若問題與資訊無關，請忽略提供的資訊並直接回答使用者問題。\n"
  f"資訊：\n```\n{document}\n```\n"
  f"使用者問題：\n```\n{user_query}\n```\n"
)
```
- 將output回傳給使用者


## Model Selection:
Embedding model: BAAI/bge-large-zh-v1.5
Reranker model: BAAI/bge-reranker-v2-m3
LLM: MediaTek-Research/Breeze-7B-Instruct-v1_0

## Current Attempt and speed testing:
- 嘗試將model轉乘onnx or tensorRT來加速運行
- 速度測試：
  - RAG: Given 512 passages, 6.34sec/per query
  - LLM: 25.54 words/per sec

## API 
```
headers = {'Content-Type': 'application/json'}
payload = {'user_message': user_message}
response = requests.post('http://pipelines.yfshih.toolmenlab.bime.ntu.edu.tw/rag', headers=headers, json=payload)
```
