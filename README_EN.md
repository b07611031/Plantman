# Plantman

## Introduction

**Plantman** combines image recognition and natural language generative AI to provide automated pest and disease identification and real-time farmer consultation services, aiming to address challenges in agricultural pest and disease detection and prevention.

The system consists of a LINE chatbot, a plant leaf pest and disease recognition model, and an agricultural pest and disease consultation system. Users can upload leaf images through Plantman's LINE chatbot. The system will automatically identify possible lesions on the leaves and their types, providing recognition results and related prevention methods. Additionally, users can consult Plantman via text to acquire knowledge about agricultural pests and diseases, gaining a comprehensive understanding of prevention steps and situations for pests, diseases, and physiological disorders through Q&A.

## Team Introduction

**Plantmen** is a team from the Machine Learning and Computer Vision Laboratory of the Department of Bio-Industrial Mechatronics Engineering at National Taiwan University. We are dedicated to applying engineering knowledge to bio-related industries to solve various challenges. Smart agriculture is a primary research direction of our lab, where we apply AI, computer vision, and natural language technologies to agriculture to address the high diversity and variability in agricultural environments.

## System Architecture and Technical Details

Plantman consists of three main modules:

1. **Controller**: Developed in Python with Flask, it communicates with the instant messaging platform LINE (frontend), controlling recognition logic and data flow between models.

2. **Image Object Detection Model**: Responsible for locating and classifying lesions on leaf images, using YOLOv9-C as the image recognition model.

3. **Natural Language Processing Model**: Combines Retrieval Augmented Generation (RAG) with a Large Language Model (LLM) to generate professional pest and disease prevention information.
   - **RAG**: Uses Reranker Based RAG with the embedding model `bge-large-zh-v1.5` and the reranker model `bge-reranker-v2-m3`.
   - **LLM**: Uses `Breeze-7B-Instruct-v1_0`.

## Data Source and Model Details

| Component                             | Model Size | License             |
|---------------------------------------|------------|---------------------|
| **Detection Model YOLOv9-C**          | 25.3M      | GPL-3.0 license     |
| **Embedding Model bge-large-zh-v1.5** | 326M       | MIT License         |
| **Reranker Model bge-reranker-v2-m3** | 568M       | Apache license 2.0  |
| **LLM Model Breeze-7B-Instruct-v1_0** | 7.49B      | Apache license 2.0  |

## Hardware

- **GPU**:
  - NVIDIA GeForce RTX 2080 Ti (11,264 MiB VRAM) ×2
  - NVIDIA GeForce A6000 (49,140 MiB VRAM) ×2
- **CPU**:
  - Intel Xeon CPU E5-2640 v3 @ 2.60GHz (16 cores)
- **Memory (RAM)**:
  - Samsung M393A4K40CB2-CTD 32GB ×8
  - **Total Memory Capacity**: 256 GB
- **Disk Storage**:
  - **Total Capacity**: 4.64 TB