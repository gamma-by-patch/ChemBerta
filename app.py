# app.py
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import huggingface_hub

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
CACHE_DIR = "./model_cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
    title="ChemBERTa Inference API",
    description="API for chemical property predictions using ChemBERTa",
    version="1.0.0"
)

class SMILESRequest(BaseModel):
    smiles: str
    max_length: int = 512

class EmbeddingResponse(BaseModel):
    embeddings: list
    device: str
    model: str

def download_model():
    """Download model with progress tracking"""
    if not os.path.exists(os.path.join(CACHE_DIR, MODEL_NAME)):
        logger.info(f"Downloading {MODEL_NAME}...")
        huggingface_hub.snapshot_download(
            MODEL_NAME,
            local_dir=CACHE_DIR,
            resume_download=True,
            cache_dir=CACHE_DIR
        )
        logger.info("Model download complete")

def load_model():
    """Load model with GPU optimization"""
    download_model()
    
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(CACHE_DIR, MODEL_NAME)
    )
    
    model = AutoModel.from_pretrained(
        os.path.join(CACHE_DIR, MODEL_NAME)
    ).to(DEVICE)
    
    # GPU optimization
    if DEVICE == "cuda":
        model = model.half()  # Use FP16 for faster inference
        torch.backends.cudnn.benchmark = True
    
    logger.info(f"Model loaded on {DEVICE.upper()}")
    return tokenizer, model

# Load during startup
tokenizer, model = load_model()

@app.on_event("startup")
async def startup_event():
    logger.info(f"API ready on {DEVICE.upper()}")
    logger.info(f"Torch version: {torch.__version__}")
    logger.info(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

@app.post("/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: SMILESRequest):
    try:
        # Tokenize input
        inputs = tokenizer(
            request.smiles,
            padding=True,
            truncation=True,
            max_length=request.max_length,
            return_tensors="pt"
        ).to(DEVICE)
        
        # Generate embeddings
        with torch.no_grad():
            if DEVICE == "cuda":
                with torch.cuda.amp.autocast():  # Mixed precision
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
        
        # Extract embeddings ([CLS] token)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()
        
        return {
            "embeddings": embeddings,
            "device": DEVICE,
            "model": MODEL_NAME
        }
    
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": DEVICE,
        "model": MODEL_NAME,
        "vrAM_used": f"{torch.cuda.memory_allocated()/1e9:.2f}GB" if DEVICE=="cuda" else "N/A"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Only 1 worker for GPU
        timeout_keep_alive=300
)
