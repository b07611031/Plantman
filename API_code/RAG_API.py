# app.py
from flask import Flask, request, jsonify
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import json
import torch
app = Flask(__name__)

# Global variables to store initialized objects
model = None
tokenizer = None
passages = None
mappings = None

ROOT  = "/app/pipelines"

def initialize_server():
    global model, tokenizer, passages, mappings
    try:
        print("Initializing server settings...")

        # Example: Load a machine learning model
        print("Loading model...")
        model = ORTModelForSequenceClassification.from_pretrained(
            f'{ROOT}/model/bge_reranker_large',
            file_name="onnx/model.onnx",
            provider='CUDAExecutionProvider'
        )
        tokenizer = AutoTokenizer.from_pretrained(f'{ROOT}/model/bge_reranker_large')
        # Example: Load documents
        print("Loading documents...")
        with open(f"{ROOT}/RAG_DATA/fullText.json", 'r', encoding='utf-8') as f:
            passages = json.load(f)
        with open(f"{ROOT}/RAG_DATA/directPairings.json", 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        # Add any other initialization tasks here
        print("Initialization complete.")

    except Exception as e:
        print(f"Error during initialization: {e}")
        import sys
        sys.exit(1)

@app.route('/process', methods=['POST'])
def process_message():
    try:
        data = request.get_json()
        if not data or 'user_message' not in data:
            return jsonify({'error': 'No user_message provided'}), 400

        user_message = data['user_message']
        print(f"Received user_message: {user_message}")

        inputs = tokenizer([[user_message, passage] for passage in passages],
                       padding=True, truncation=True, max_length=512, return_tensors='pt')

        # Example: Use loaded documents if needed
        # result = some_function(processed_message, documents)

        # For demonstration, let's create a RAG passage
        with torch.inference_mode():
            scores = model(**inputs).logits.view(-1).float()
        
        top_indices = torch.argsort(scores, descending=True).tolist()

        return jsonify({'rag_passage': top_indices}), 200

    except Exception as e:
        print(f"Error processing message: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Call the initialization function before starting the server
initialize_server()

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)

