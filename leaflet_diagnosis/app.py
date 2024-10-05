from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
from PIL import Image
import numpy as np
from leaflet_diagnosis_model_Allen import inference_leaflet_diagnosis_model  # Import your model inference function

app = FastAPI()

# Root welcome message
@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI YOLOv9 prediction API!"}

# Define an endpoint for image upload and inference
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / file.filename
    
    with temp_file.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # image = Image.open(temp_file)
    # image_matrix = np.array(image).tolist()
    
    # image_path = "/home/nas/Research_Group/Personal/Yun/Tomato/LINEBot/0.0.4/static/default/disorders/Phytohormone_damage.jpg"

    # Run inference using the saved file
    prediction, bbs = inference_leaflet_diagnosis_model(str(temp_file))
    
    # prediction = inference_leaflet_diagnosis_model(image_path)
    
    # Clean up temp file after inference
    temp_file.unlink()
    
    return {"bounding_boxs": bbs, "predictions": prediction}

# To run the app using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)