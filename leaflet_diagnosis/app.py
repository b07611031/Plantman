from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
from typing import List
from PIL import Image
import numpy as np
from leaflet_diagnosis_model_Allen import inference_leaflet_diagnosis_model  # Import your model inference function
from cv2 import imwrite

app = FastAPI()

STATIC_DIR = Path("./static/url_images")
STATIC_DIR.mkdir(parents=True, exist_ok=True)

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
        
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            

        # Run inference using the saved file
        prediction, bbs, img = inference_leaflet_diagnosis_model(str(temp_file))
        image_save_path = STATIC_DIR / f"{file.filename}"  # 使用上傳的檔名作為保存名
        # img.save(image_save_path)
        imwrite(str(image_save_path), img)
        image_url = f"http://127.0.0.1:8080/static/url_images/{file.filename}"
        
        # prediction = inference_leaflet_diagnosis_model(image_path)
        
        # Clean up temp file after inference
        temp_file.unlink()
        
        result.append({"image_name":file.filename, "bounding_boxs": bbs, "predictions": prediction, "image_url": image_url})
        
    return {"results": result}

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

# To run the app using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)