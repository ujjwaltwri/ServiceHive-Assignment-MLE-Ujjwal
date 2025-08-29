Technical Report: Intelligent Scene Analysis System
Author: Ujjwal Tiwari
Date: August 29, 2025

1. Introduction
1.1. Project Goal

The primary objective of this project was to build and deploy an intelligent service capable of classifying natural scene images into six distinct categories, generating an appropriate text description, and providing confidence and uncertainty scores for its predictions. The entire system was required to be delivered as a containerized REST API, demonstrating an end-to-end understanding of the MLOps lifecycle.

1.2. Dataset

The project utilized the "Intel Image Classification" dataset from Kaggle. This dataset contains approximately 25,000 JPEG images of size 150x150 pixels, balanced across six classes: building, forest, glacier, mountain, sea, and street.

2. Model Development and Training
Two primary modeling approaches were undertaken as required by the brief: building a custom CNN from scratch and implementing a high-performance transfer learning solution.

2.1. Model 1: Custom CNN (Baseline)

Architecture:
A sequential CNN was constructed with three convolutional blocks, each consisting of a Conv2D layer (ReLU activation) followed by a MaxPooling2D layer. The number of filters increased with depth (32, 64, 128). The output was flattened and passed through a Dense layer (128 units) with Dropout (0.5) for regularization, and a final softmax output layer for the 6 classes.

Performance:
The custom CNN achieved a final validation accuracy of ~77%. While this successfully met the initial project requirement of ≥75% validation accuracy, analysis of the training curves revealed significant overfitting, indicating a clear need for a more robust modeling strategy.

2.2. Model 2: Transfer Learning with EfficientNetV2B0 (Final Production Model)

Architecture and Rationale:
To surpass the 77% accuracy mark and meet the ≥85% target, a transfer learning approach was adopted. The EfficientNetV2B0 architecture, pre-trained on ImageNet, was chosen for its high efficiency and state-of-the-art performance. The base model's layers were frozen, and a new classification head was added on top, consisting of GlobalAveragePooling2D, BatchNormalization, and the final softmax classifier. The input image size was increased to 224x224 to match the model's preferred input resolution.

Training Strategy:
A more robust training strategy was employed to combat overfitting and meet the "advanced trick" requirement:

Data Augmentation: RandomFlip, RandomRotation, RandomZoom, and RandomContrast were applied to the training dataset to increase its diversity.

EarlyStopping Callback: This advanced trick was used to monitor val_accuracy with a patience of 5 epochs. It automatically halted training when performance plateaued, ensuring the model with the best weights was retained and preventing further overfitting.

Performance:
This approach was highly successful. The model achieved a peak validation accuracy of ~92.8%, comfortably exceeding the project requirement of ≥85%.

3. Multimodal Integration
3.1. Vision-Text Pipeline: Scene Description Generation

The final predicted class from the EfficientNetV2B0 model is used to dynamically generate a prompt (e.g., "Write a single, evocative, one-sentence description for an image of a forest scene."). This prompt is sent to the Google Gemini 1.5 Flash API, which returns a contextually appropriate text description, fulfilling the vision-text pipeline requirement.

3.2. Uncertainty Estimation: Predictive Entropy

The initial plan was to use Monte-Carlo Dropout for uncertainty. However, persistent and deep-seated bugs related to the serialization of the model's Dropout layer made this approach unreliable, always resulting in artificially low confidence scores.

To solve this problem innovatively, the final implementation uses a more robust and mathematically sound uncertainty metric: Predictive Entropy. This is calculated directly from the model's final softmax probability distribution. A highly confident prediction (e.g., [0.99, 0.01, ...]) has very low entropy (low uncertainty), while a less confident prediction (e.g., [0.4, 0.3, ...]) has high entropy (high uncertainty). This method is deterministic, fast, and satisfies the requirement for an uncertainty metric.

4. System Architecture and Deployment
The final application is a robust, containerized web service.

API: A FastAPI server exposes the functionality through three required endpoints: /health, /model_info, and /predict. CORS middleware is enabled to allow the frontend to interact with the API.

Model Serving: The trained EfficientNetV2B0 model and the configured Gemini model are loaded into memory once at startup using FastAPI's lifespan manager for efficient inference.

Containerization: The entire application is packaged into a Docker image using a Dockerfile, with a docker-compose.yml file provided for easy, one-command deployment. A .dockerignore file is used to ensure a minimal and efficient build.

Testing: The system includes a suite of unit tests (pytest) that verify the functionality of the API endpoints, ensuring engineering quality and reliability.

5. Conclusion
This project successfully met all technical requirements outlined in the assessment brief. A high-accuracy classification model was developed and integrated into a full-featured API, complete with multimodal capabilities and a professional deployment strategy. The final system is reliable, efficient, and demonstrates a comprehensive understanding of the end-to-end

