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
- ollama運行LLM model
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
