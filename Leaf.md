# Leaf Diagnosis Model

## Project details
- Programming Language

Python

- Library Used
PyTorch for deep learning model development, ONNX for model export and inference, NumPy for data manipulation, and OpenCV for image preprocessing.

- GPU Programming Model

CUDA for GPU acceleration, utilizing cuDNN for deep learning optimizations. Also leveraging PyTorch’s built-in GPU support for tensor computations. The ONNX model inference is optimized using TensorRT for deployment on the GPU.

- Frameworks and Models Used for Data

The primary model used is based on a modified YOLOv9 architecture, fine-tuned for detecting multiple plant leaf disorders. The optimizer used is Adam with a learning rate scheduler. We’ve worked on cloud workstations and GPUs for training, using custom plant disease datasets. 

- Algorithmic Motifs

Our project primarily revolves around convolutional neural networks (CNNs) for object detection and classification. The focus is on image processing, data augmentation, and bounding box regression.

- Current Performance

The model currently runs on an NVIDIA RTX A6000 GPU for training, and inference has been also tested on an NVIDIA RTX A600-. The application scales well up to a single node with a batch size of 16 images for training. Current inference times average around 50 milliseconds per image when using ONNX on the GPU.

- Expected Outcome

- Data Source and Model Details:
The dataset used contains images of leaves from (***Yun***), with annotations for 13 types of diseases. The dataset size is approximately . The model is around 25 MB when exported to ONNX format. Training currently takes about (***Yun***) hours for 50 epochs on the NVIDIA RTX A6000 (***Yun?***) GPU.

- Computing Facilities

The model is developed and trained on a cloud high-performance workstation with an NVIDIA RTX A6000 GPU.

- License

The project is released under the MIT License, ensuring that it remains open source and accessible for further research and development in the agricultural sector.
