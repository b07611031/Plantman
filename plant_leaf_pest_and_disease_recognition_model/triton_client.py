"""
Triton Inference Client for Leaf Disease Classification (In Development)

DISCLAIMER:
This script is currently under development and cannot be used for production or reliable inference yet.
The functionality is incomplete, and it may contain issues or inefficiencies that will be addressed in future revisions.

Current Features:
- Connects to a remote Triton Inference Server.
- Preprocesses images for inference using a YOLO model.
- Attempts to retrieve and process inference results.
- Draws bounding boxes and labels on the images based on inference results.

Known Issues:
- Incomplete error handling during Triton server connection and inference.
- Some sections are not optimized or tested for various environments.
- Post-processing of inference results is in progress and may not work correctly.
- Logging and monitoring of inference processes is limited.

Planned Improvements:
- Add detailed error handling and fallback mechanisms.
- Optimize the preprocessing and inference steps for performance.
- Extend support for additional model types and inference tasks.
- Refine the output visualization to handle complex results.

NOTE:
Please use this script only in a controlled environment for testing purposes. It should not be used for any critical applications until the development is complete.
"""

import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import cv2 as cv
import numpy as np
import numba
import json

input_size = 640

class_name = {0:'Specks', 1:'Ring spots', 2:'Yellow halo spots', 3:'White powder', 4:'Yellow spots',
                  5:'Water-soaked lesion', 6:'Virus', 7:'Leaf miner fly', 8:'Leaf miner moth',
                  9:'Two-spotted spider mite', 10:'Lepidoptera', 11:'Frostbite patches', 
                  12:'Sunscald patches'}


inf_transforms = transforms.Compose(
    [
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
@numba.jit
def extractFeature(inferenceResult:np.ndarray) -> np.ndarray:
    threshold = 0.95
    final = []
    for i in inferenceResult:
        temp = []
        box = i[:4]
        confidence = i[4]
        if np.max(confidence) < threshold:
            continue
        else:
            for j in box:
                temp.append(j)
            classIndex = np.where(confidence == np.max(confidence))[0][0]
            temp.append(classIndex)
            final.append(temp)
    return final
      
      
@numba.jit
def xywh2xyxy(box:np.ndarray) -> np.ndarray:
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return np.array([x1, y1, x2, y2])

@numba.jit
def drawBox(image:np.ndarray, result:np.ndarray) -> np.ndarray:
    for i in result:
        # convert [x, y, w, h] to [x1, y1, x2, y2]
        box = xywh2xyxy(i[:4])
        classIndex = class_name[int(i[4])]
        color = (0, 255, 0)
        # draw box
        cv.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        # draw text
        cv.putText(image, classIndex, (int(box[0]), int(box[1])), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image


def image_preprocess(img_path):
    img = default_loader(path=img_path)
    img = inf_transforms(img)
    return img.unsqueeze(0).numpy()


def triton_inference(img_path):
    server_url = "triton.model.repository.toolmenlab.bime.ntu.edu.tw"
    triton_client = httpclient.InferenceServerClient(url=server_url, verbose=False)

    img = image_preprocess(img_path)
    inputs = []

    # httpclient.InferInput('input', img.shape, 'FP32').set_data_from_numpy()
    inputs.append(httpclient.InferInput("input", img.shape, "FP32"))
    inputs[0].set_data_from_numpy(img)

    outputs = []
    outputs.append(httpclient.InferRequestedOutput("output"))

    try:
        resp = triton_client.infer(
            model_name="leaf_classification_model",
            inputs=inputs,
            model_version='2',
            request_id=str(1),
            outputs=outputs,
        )
    except InferenceServerException as e:
        print("inference failed:", e)
        
    result = resp.get_result().as_numpy("output")[0].T
    return result
    

import os
import numpy as np
import time

if __name__ == "__main__":
    
    dir_path = 'static/disorders/'
    print('Start time', time.asctime(time.localtime(time.time())), '\n')
    for file_path in os.listdir(dir_path):
        if not file_path.endswith('.jpg'):
            continue
        print(file_path)
        img_path = os.path.join(dir_path, file_path)
        result = triton_inference((img_path))
        last_layer = result.as_numpy("output")
        
        print('End time', time.asctime(time.localtime(time.time())))
        print(last_layer)
        print(class_name[np.argmax(last_layer)], '\n')
        