from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from contextlib import asynccontextmanager
import google.generativeai as genai
import os

# --- Configuration ---
MODEL_PATH = 'models/efficientnet_v2.keras'
IMG_SIZE = (224, 224)
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup."""
    try:
        # --- DEFINITIVE FIX 1: Load the model in a clean inference state ---
        ml_models["classifier"] = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("Image classification model loaded in inference mode.")
        genai.configure(api_key=GEMINI_API_KEY)
        ml_models["llm"] = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Generative model configured.")
    except Exception as e:
        logger.error(f"Error during model loading: {e}")
    yield
    ml_models.clear()

app = FastAPI(title="Intelligent Scene Analysis API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---
def preprocess_image(image_bytes: bytes) -> tf.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize(IMG_SIZE)
    return tf.expand_dims(tf.keras.utils.img_to_array(img), 0)

# --- DEFINITIVE FIX 2: A new, robust analysis function ---
def get_final_analysis(model, img_array):
    """
    Performs a single, standard prediction and calculates uncertainty from the result.
    """
    prediction = model.predict(img_array, verbose=0)
    scores = prediction[0]
    
    # Calculate confidence
    confidence = float(100 * np.max(scores))
    predicted_class = CLASS_NAMES[np.argmax(scores)]
    
    # Calculate uncertainty (Predictive Entropy)
    # This measures the "evenness" of the probability distribution.
    # A high confidence (peaked distribution) results in low entropy (low uncertainty).
    scores = scores[scores > 0] # Avoid log(0) for stability
    entropy = -np.sum(scores * np.log2(scores))
    
    return predicted_class, confidence, float(entropy)

def get_scene_description(llm, class_name: str) -> str:
    try:
        prompt = f"Write a single, evocative, one-sentence description for an image of a {class_name} scene."
        response = llm.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"LLM description generation failed: {e}")
        return f"A beautiful scene of a {class_name}."

# --- API Endpoints ---
@app.get("/", response_class=FileResponse, include_in_schema=False)
def read_root():
    return "frontend/index.html"

@app.get("/health", summary="Check if the service is running")
def health_check():
    models_loaded = ml_models.get("classifier") is not None and ml_models.get("llm") is not None
    return {"status": "ok", "models_loaded": models_loaded}

@app.get("/model_info", summary="Get information about the ML model")
def model_info():
    if ml_models.get("classifier"):
        return {"model_name": "EfficientNetV2B0", "input_shape": IMG_SIZE, "class_names": CLASS_NAMES}
    raise HTTPException(status_code=503, detail="Model is not available.")

@app.post("/predict", summary="Classify an image and generate a description")
async def predict(file: UploadFile = File(...)):
    classifier = ml_models.get("classifier")
    llm = ml_models.get("llm")
    if not classifier or not llm:
        raise HTTPException(status_code=503, detail="A model is not available.")
    image_bytes = await file.read()
    try:
        img_array = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    
    pred_class, conf, uncertainty_score = get_final_analysis(classifier, img_array)
    description = get_scene_description(llm, pred_class)

    logger.info(f"Full analysis for {file.filename} complete.")
    
    return JSONResponse(content={
        "filename": file.filename,
        "prediction": pred_class,
        "confidence": conf,
        "uncertainty": uncertainty_score,
        "description": description
    })