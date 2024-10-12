# YOLOv9: Object Detection and Classification

YOLOv9 is a real-time object detection and classification model, built to enhance the YOLO (You Only Look Once) family of models, specifically designed for high accuracy and performance. It can be used for various tasks such as detecting objects in images and videos, classifying objects, and segmenting images.

## Features
- **Real-time Performance:** Optimized for high-speed inference on GPUs.
- **High Accuracy:** Built on top of YOLOv9 architecture with custom tweaks for accuracy.
- **ONNX Support:** Export and inference through ONNX Runtime.
- **Multi-GPU Support:** Seamlessly scales across multiple GPUs.
- **Flexible Input Sizes:** Supports input image resolutions of 640x640, 1280x1280, and more.
- **Pretrained Models:** Access to pretrained weights for various datasets like COCO, VOC, and custom datasets.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Exporting to ONNX](#exporting-to-onnx)
- [Performance](#performance)
- [License](#license)

## Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- PyTorch 1.10+ with GPU support
- CUDA 10.2+
- Torchvision
- OpenCV

Install the required Python libraries:
```bash
pipenv install
pipenv shell
```


## Inference
```
python detect_dual.py --source /path/to/image.jpg --data data/data.yaml --weights yolov9.pt --device 0 --save-txt --save-conf
```

- --source: Path to the image or video file for inference.
- --weights: Path to the trained weights file. Could be PyTorch or ONNX
- --img-size: Input image size.

## Exporting to ONNX

YOLOv9 supports exporting the model to ONNX for fast inference with ONNX Runtime
```
python export.py --weights yolov9.pt --include onnx --data your_data_path.yaml
```

## Performance
Inference Speed: <35ms per image on NVIDIA RTX A6000 (640x640 resolution).

