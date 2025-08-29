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

Dynamic Scene Description: Integrates with the Google Gemini 1.5 Flash API to generate unique, context-aware descriptions for each image prediction.

Uncertainty Estimation: Implements Monte-Carlo Dropout to provide a quantifiable uncertainty score for each prediction.

RESTful API: A robust API built with FastAPI featuring /health, /model_info, and /predict endpoints.

Containerized Deployment: Fully containerized using Docker and Docker Compose for consistent and reliable deployment.

Interactive Frontend: A polished HTML/JavaScript frontend for easy testing and demonstration of the API's capabilities.

3. Tech Stack
Backend: Python, FastAPI

ML/DL: TensorFlow, Keras, Scikit-learn, Pandas

Generative AI: Google Gemini 1.5 Flash API

Deployment: Docker, Docker Compose, Uvicorn

Frontend: HTML, CSS, JavaScript

4. Setup and Installation
Prerequisites

Python 3.9+

Docker and Docker Compose

A Google Gemini API Key

Instructions

Clone the Repository:

git clone [https://github.com/your-username/ServiceHive-Assignment-MLE-Ujjwal.git](https://github.com/your-username/ServiceHive-Assignment-MLE-Ujjwal.git)
cd ServiceHive-Assignment-MLE-Ujjwal

Set Up Environment Variables:

Create a file named .env in the main project directory.

Add your Gemini API key to this file:

GEMINI_API_KEY=YOUR_API_KEY_HERE

(Optional) Local Python Environment:

Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate

Install dependencies from the provided requirements.txt file:

pip install -r requirements.txt

5. Running the Application
Method 1: Docker (Recommended)

This is the simplest way to run the entire application. It builds the Docker image and starts the container.

docker compose up --build

The API will be available at http://127.0.0.1:8000.

Method 2: Local Development Server

Make sure your virtual environment is activated and your .env file is created.

python -m uvicorn SIC.api.main:app --reload --reload-dir SIC

6. Usage
Interactive Frontend

Once the application is running, open your web browser and navigate to the root URL to use the interactive test page:
http://127.0.0.1:8000

API Documentation

FastAPI provides automatic interactive documentation (Swagger UI). To access it, go to:
http://127.0.0.1:8000/docs

You can test all the API endpoints directly from this page, including file uploads to the /predict endpoint.

7. Running Tests
To run the automated unit tests, ensure your virtual environment is active and run pytest from the main project directory:

python -m pytest
