from flask import Flask, request
import configparser
import time
import os
import requests
import base64
import io
from PIL import Image

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
    ImageMessage,
    ImageSendMessage,
    VideoMessage,
    AudioMessage,
    FileMessage,
    LocationMessage,
    StickerMessage,
    FlexSendMessage,
    PostbackEvent,
    QuickReply,
    QuickReplyButton,
    CameraAction,
    CameraRollAction,
    MessageAction,
)


cf = configparser.ConfigParser()
cf.read("./controller/config.ini")
line_bot_api = LineBotApi(cf.get("line-developer", "api"))
handler = WebhookHandler(cf.get("line-developer", "handler"))
url_basis = cf.get("line-developer", "url_basis")
url_leaflet_diagnosis_model = cf.get("api-url", "leaflet_diagnosis_model")
url_rag_reranker_llm = cf.get("api-url", "rag_reranker_llm")

app = Flask(__name__)

def handle_leaflet_diagnosis_model_api(image_path):
    url_leaflet_diagnosis_model = "http://cv.plantman.toolmenlab.bime.ntu.edu.tw/predict/"

    # Open the file in binary mode and send it with the request
    with open(image_path, 'rb') as img:
        files = {'files': img}  # 'files' matches the parameter in your FastAPI endpoint
        try:
            response = requests.post(url_leaflet_diagnosis_model, files=files)
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred on leaflet diagnosis model: {http_err}")
            print("伺服器發生錯誤，請稍後並重新上傳資料。")
            return {}
        except Exception as err:
            print(f"An error occurred on leaflet diagnosis model: {err}")
            print("出現了未知錯誤，請稍後再試。")
            return {}
            
        # Process the response and decode the image
        response_image_base64 = response.json()['results'][0]['image']
        response_image_data = base64.b64decode(response_image_base64)
        img = Image.open(io.BytesIO(response_image_data))

        # Save the predicted image
        predicted_image_name = f'prediction_{os.path.basename(image_path)}'
        # Use os.path.dirname to get the directory of the original image
        predicted_image_path = os.path.join(os.path.dirname(image_path), predicted_image_name)
        
        img.save(predicted_image_path)
        return predicted_image_name

@app.route("/", methods=["POST", "GET"])
def callback():
    # get X-Line-Signature header value
    signature = request.headers.get("X-Line-Signature", None)

    # get request body as text
    body = request.get_data(as_text=True)
    print("Request body: " + body)

    # Skip signature verification for testing if the header is missing
    if signature is None:
        print("Skipping signature verification for testing purposes.")
    else:
        # handle webhook body, make sure the message was from LINE by using the channel secret
        try:
            handler.handle(body, signature)
        except InvalidSignatureError:
            print(
                "Invalid signature. Please check your channel access token/channel secret."
            )
            os.abort(400)  # Return HTTP 400 Bad Request if the signature is invalid

    return "OK"


# handle Message Event / Text message
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    if event.source.user_id == "Udeadbeefdeadbeefdeadbeefdeadbeef":
        return
    print("User ID:", event.source.user_id)
    print("User name:", line_bot_api.get_profile(event.source.user_id).display_name)
    print("User text:", event.message.text)
    data = {"user_message": event.message.text}
    try:
        response = requests.post(
            url_rag_reranker_llm,
            headers={"Content-Type": "application/json"},
            json=data,
        )
        response.raise_for_status()
        response_message = response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        response_message = "伺服器發生錯誤，請稍後並重新上傳資料。"
    except Exception as err:
        print(f"An error occurred: {err}")
        response_message = "出現了未知錯誤，請稍後再試。"

    line_bot_api.reply_message(
        event.reply_token, TextSendMessage(text=response_message)
    )


# handle Message Event / Image message
@handler.add(MessageEvent, message=ImageMessage)
def handle_message(event):
    if event.source.user_id == "Udeadbeefdeadbeefdeadbeefdeadbeef":
        return
    # get image from LINE Server
    message_content = line_bot_api.get_message_content(event.message.id)
    received_time = "{}".format(time.time())
    # user_name = line_bot_api.get_profile(event.source.user_id).display_name
    # filename = f"{user_name}_{received_time}.jpg"
    filename = f"{received_time}.jpg"
    image_path = os.path.join("./static", filename)
    # save image to local
    with open(image_path, "wb") as img:
        for chunk in message_content.iter_content():
            img.write(chunk)
    
    # 回覆可以優化，提供flex image，並在沒有偵測到的時候提醒使用者。
    predicted_image_name = handle_leaflet_diagnosis_model_api(image_path)
    if predicted_image_name:
        # reply image message by using LINE message API
        line_bot_api.reply_message(
            event.reply_token,
            ImageSendMessage(
                original_content_url=url_basis + predicted_image_name,
                preview_image_url=url_basis + predicted_image_name,
            ),
        )
    else:
        response_message = "影像辨識模型出現了未知錯誤，請稍後再試。"
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text=response_message)
        )


if __name__ == "__main__":
    # run the server at localhost:5000
    app.run(debug=False, host="0.0.0.0", port=5000)
