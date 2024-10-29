# LINEBOT Controller

## Introduction
The controller is developed in Python with Flask and is used to communicate with the instant messaging platform (frontend), controlling recognition logic and data flow between models.

## Installation

<details><summary> <b>Expand</b> </summary>

``` shell
# pip install required packages
pip install -r requirements.txt

# edit your linebot information
vi config.ini

# run the controller
python controller.py
```

</details>

## Docker Image

### Dockerfile Details

The Dockerfile used to create this image is as follows:

```
# Use the official Python 3.8 image based on Ubuntu 20.04 as the base image
FROM python:3.8-slim-buster

# Set environment variables to prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Install any dependencies required for Pillow or other libraries
RUN apt-get update && \
    apt-get install -y libjpeg-dev zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements to the working directory
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . /app

# Add the config.ini file from a URL or local file
ADD config.ini /app

# Expose port 5000 for external access
EXPOSE 5000

# Set the default command to start the application
CMD ["python", "LINEBOT_v3.py"]

```
*config.ini*
```
[line-developer]
# channel access token
api = <your linebot token>

# channel secret
handler = <your linebot secret>

# channel https link
url_basis = <your link>/static
# url_basis example
url_basis = https://plantman.toolmenlab.bime.ntu.edu.tw/static/

[api-url]
# diagnosis model link
leaflet_diagnosis_model = http://cv.plantman.toolmenlab.bime.ntu.edu.tw/predict/

# RAG link
rag_reranker_llm = http://pipelines.yfshih.toolmenlab.bime.ntu.edu.tw/rag
```
### Running the Container

To run the container and map port `5000` on the container to port `5000` on the host, use the following command:

```bash
docker run -p 5000:5000 b07611031/plantman_controller:0.0.0

```

### Accessing the Application

Once the container is running, you can access the application from your host machine by opening your browser and navigating to:

```bash
http://localhost:5000
```

If everything is set up correctly, you should see the application running on this URL.
