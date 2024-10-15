# app.py
from flask import Flask, request, jsonify
import json
import requests
import sys
from waitress import serve
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Global variables to store initialized objects
passages = None
mappings = None

ROOT = "/app/pipelines"

def initialize_server():
    global passages, mappings
    try:
        print("Initializing server settings...")

        print("Loading documents...")
        with open(f"{ROOT}/RAG_DATA/fullText.json", 'r', encoding='utf-8') as f:
            passages = json.load(f)
        with open(f"{ROOT}/RAG_DATA/directPairings.json", 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        # Add any other initialization tasks here
        print("Initialization complete.")

    except Exception as e:
        print(f"Error during initialization: {e}")
        sys.exit(1)

@app.route('/rag', methods=['POST'])
def process_message():
    try:
        data = request.get_json()
        if not data or 'user_message' not in data:
            return jsonify({'error': 'No user_message provided'}), 400

        user_message = data['user_message']
        print(f"Received user_message: {user_message}")
        headers = {
            'Content-Type': 'application/json'
        }
        payload = {
            'user_message': user_message
        }
        try:
            response = requests.post('http://localhost:8080/process', headers=headers, json=payload)
            response.raise_for_status()  # Raises stored HTTPError, if one occurred.

            # Parse the JSON response
            response_data = response.json()
            indices = list(response_data['rag_passage'])
            seleected_passage = passages[int(indices[0])]
            seleected_passage = seleected_passage.replace("~", " ~ ")
            prompt = (
                f"請根據以下資訊回答使用者的問題。若問題與資訊無關，請忽略提供的資訊並直接回答使用者問題。\n"
                f"資訊：\n```\n{seleected_passage}\n```\n"
                f"使用者問題：\n```\n{user_message}\n```\n"
            )
            body = {
                "messages":[{"role": "user", "content": prompt}]
                }
            try:
                response = requests.post(
                    headers={"Content-Type": "application/json"},
                    url=f"http://ollama.chris.toolmenlab.bime.ntu.edu.tw/v1/chat/completions",
                    json={**body, "model": "Breeze_7B_Instruct_v1_0"},
                    stream=False,
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                return f"Error: {e}"

        except requests.exceptions.HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')  # HTTP error
        except requests.exceptions.ConnectionError as conn_err:
            print(f'Connection error occurred: {conn_err}')  # Connection error
        except requests.exceptions.Timeout as timeout_err:
            print(f'Timeout error occurred: {timeout_err}')  # Timeout error
        except requests.exceptions.RequestException as req_err:
            print(f'An error occurred: {req_err}')  # Any other request errors
        except json.JSONDecodeError:
            print('Failed to parse JSON response.')
        

        return None, 500
    except Exception as e:
        print(f"Error processing message: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Call the initialization function before starting the server
initialize_server()

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=9090)

