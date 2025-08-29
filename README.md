Intelligent Scene Analysis System
Author: Ujjwal Tiwari
Project for: ServiceHive AI/ML Intern Assessment

1. Project Overview
This project is an end-to-end machine learning application that provides a complete intelligent scene analysis service. As per the assignment brief, the system is designed to perform three main tasks:

Classify natural scene images into one of six categories: buildings, forest, glacier, mountain, sea, or street.

Generate a short, scene-appropriate text description for each prediction.

Provide a confidence score and an uncertainty estimate for the classification.

The entire service is exposed via a REST API, built with FastAPI, and is fully containerized with Docker for easy deployment and reproducibility.

2. Features
High-Accuracy Image Classification: Utilizes a fine-tuned EfficientNetV2B0 model achieving ~92.8% validation accuracy, surpassing the ≥85% requirement.

Dynamic Scene Description: Integrates with the Google Gemini 1.5 Flash API to generate unique, context-aware descriptions.

Uncertainty Estimation: Implements Predictive Entropy, a robust uncertainty metric calculated from the model's softmax output.

RESTful API: A robust API built with FastAPI featuring /health, /model_info, and /predict endpoints.

Containerized Deployment: Fully containerized using Docker and Docker Compose for consistent and reliable deployment.

Interactive Frontend: A polished HTML/JavaScript frontend for easy testing and demonstration of the API's capabilities.

3. Tech Stack
Backend: Python, FastAPI

ML/DL: TensorFlow, Keras, Scikit-learn, Pandas

Generative AI: Google Gemini 1.5 Flash API

Deployment: Docker, Docker Compose, Uvicorn

Frontend: HTML, CSS, JavaScript

4. Project Structure
The project follows the directory structure outlined in the assignment brief:

├── Dockerfile
├── README.md
├── SIC/
│   └── api/
│       └── main.py
├── data/
│   └── (Downloaded Dataset)
├── docs/
│   └── technical_report.md
├── frontend/
│   └── index.html
├── models/
│   └── efficientnet_final_model.keras
├── notebooks/
│   ├── eda.ipynb
│   ├── training.ipynb
│   └── evaluation.ipynb
├── requirements.txt
└── tests/
    └── test_api.py

5. Setup and Running the Application
Prerequisites

Docker and Docker Compose

A Google Gemini API Key

Instructions

Clone the Repository:

# Replace with your final repository URL
git clone [https://github.com/ujjwaltwri/ServiceHive-Assignment-MLE-Ujjwal.git](https://github.com/ujjwaltwri/ServiceHive-Assignment-MLE-Ujjwal.git)
cd ServiceHive-Assignment-MLE-Ujjwal

Set Up Environment Variables:

Create a file named .env in the main project directory.

Add your Gemini API key to this file:

GEMINI_API_KEY=YOUR_API_KEY_HERE

Run with Docker:
This is the simplest way to run the entire application.

docker compose up --build

The API will be available at http://12_7.0.0.1:8000.

6. Usage
Interactive Frontend

Once the application is running, open your web browser and navigate to the root URL to use the interactive test page:
http://127.0.0.1:8000

API Documentation

FastAPI provides automatic interactive documentation (Swagger UI). To access it, go to:
http://127.0.0.1:8000/docs

7. Running Tests
To run the automated unit tests, you must first set up a local Python environment.

# Set up and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run pytest
python -m pytest

8. Citations
Dataset: Puneet Singh, "Intel Image Classification", Kaggle, Version 2, https://www.kaggle.com/datasets/puneet6060/intel-image-classification

Pre-trained Model: The EfficientNetV2B0 model architecture with weights pre-trained on the ImageNet dataset was sourced from the Keras Applications library within TensorFlow.

