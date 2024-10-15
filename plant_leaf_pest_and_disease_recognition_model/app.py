"""
FastAPI-based YOLOv9 Image Prediction API

This FastAPI application allows users to upload one or multiple images, run inference using the `inference_leaflet_diagnosis_model`, 
and return the predictions along with bounding boxes and a URL to the processed image.

"""

from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
from typing import List
from PIL import Image
import numpy as np
from leaflet_diagnosis_model import inference_leaflet_diagnosis_model  # Import your model inference function
from cv2 import imwrite

import os

app = FastAPI()

# STATIC_DIR = Path("./static/url_images")
# STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Root welcome message
@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI YOLOv9 prediction API!"}

# Define an endpoint for image upload and inference
@app.post("/predict/")
async def predict(files: List[UploadFile] = File(...)):
    result = []
    for file in files:
        # Save the uploaded file to a temporary location
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / file.filename
        
        with temp_file.open("wb") as buffer:  # "wb" for write binary
            shutil.copyfileobj(file.file, buffer)
            

        # Run inference using the saved file
        prediction, bbs, img_base64 = inference_leaflet_diagnosis_model(str(temp_file))
        
        
        # Save the processed image to a static directory (For Development)
        # image_save_path = STATIC_DIR / f"{file.filename}"  # 使用上傳的檔名作為保存名
        
        # image_filename = os.path.splitext(file.filename)[0]
        # image_url = f"http://127.0.0.1:5000/static/url_images/{file.filename}"
        # with open(f"static/url_images/{image_filename}.txt", "w") as txt_file:
        #     txt_file.write(img_base64)
       
       
        # Clean up temp file after inference
        temp_file.unlink()  # 相當於 os.remove(temp_file) 或是系統的 rm temp_file
        result.append({"image_name":file.filename, "bounding_boxs": bbs, "predictions": prediction, "image": img_base64})
        
    return {"results": result}

# from fastapi.staticfiles import StaticFiles
# app.mount("/static", StaticFiles(directory="static"), name="static")

# To run the app using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)