# Plant Leaf Pest and Disease Recognition Model

## 模型選擇：

影像物件偵測模型：finetuned yolov9

## 安裝要求
請確保已安裝以下內容：

- Python 3.8+
- PyTorch 2.4.1+（支援GPU）
- CUDA 12.1+
- Torchvision 0.19.1+
- OpenCV 4.10.0+

## 目前嘗試與速度測試
- 嘗試將模型轉換為 ONNX 或 TensorRT ，並結合Triton service來加速運行。
- 速度測試結果：
  Inference times average around 50 milliseconds per image when using ONNX on one GPU (NVIDIA GeForce A6000). While the GPU utilization can reach nearly 100% during the warm-up phase, in the detection inference phase, only one API call can be processed at a time, and each call only utilizes 20% of the GPU, leaving 80% of the GPU underutilized. The model's computation time could be significantly reduced if the GPU could be utilized more efficiently.

## API service 設置：

### 環境準備：

1. 安裝套件：
    ```bash
    pipenv install
    ```

2. 開啟環境：
    ```bash
    pipenv shell
    ```
### 執行API Service：
於 `Tomato-DPD-Identification-System/leaflet_diagnosis/` 執行，

```bash
python app.py
```

### 圖片上傳與預測：
- 使用 `/predict/` 路由進行圖片上傳，該路由接受多個圖片文件作為輸入。
- 圖片文件會先保存至臨時目錄 ./temp_uploads，以便進行處理。
- 上傳的每個圖片會通過 `inference_leaflet_diagnosis_model` 函數進行推論，返回預測結果、邊界框資訊和經過推論後的 base64 格式圖片。
- 處理後的 base64 格式圖片、邊界框及預測結果會作為 JSON 格式回傳給使用者。
- 預測完成後，臨時保存的圖片文件會自動刪除，釋放資源。

## API 範例

### API Endpoint
- **URL**: `http://cv.plantman.toolmenlab.bime.ntu.edu.tw/predict/`
- **HTTP Method**: `POST`
- **Content Type**: `multipart/form-data`

### Request Body
- **files**: The image file(s) to be uploaded. Must be sent as `multipart/form-data`.
  - Example: 
    ```python
    files = {'files': open('your_image.jpg', 'rb')}
    ```
  - **Supported formats**: JPG, PNG

### Response Body
The API returns a JSON object with the following structure:

```json
{
  "results": [
    {
      "image_name": "your_image.jpg",
      "bounding_boxes": [...],  # List of bounding boxes for identified objects
      "predictions": [...],  # List of predicted classes or labels
      "image": "base64_encoded_image_data"  # Processed image encoded in Base64
    }
  ]
}
```

### Example Python Code
```Python
import requests
import os
import base64
import io
from PIL import Image

# The API endpoint
url = "http://cv.plantman.toolmenlab.bime.ntu.edu.tw/predict/"

# The path to the image you want to upload
image_path = "your_image.jpg"

# Open the file in binary mode and send it with the request
with open(image_path, 'rb') as img:
    files = {'files': img}  # 'files' matches the parameter in your FastAPI endpoint
    response = requests.post(url, files=files)

    # Check for errors
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)  # Print error message from server
    else:
        # Save the processed image 
        filename = 'your_identified_image.jpg'
        img_base64 = response.json()['results'][0]['image']
        img_data = base64.b64decode(img_base64)
        img = Image.open(io.BytesIO(img_data))
        img.save(filename)
        print(f"Processed image saved as {filename}")
```
