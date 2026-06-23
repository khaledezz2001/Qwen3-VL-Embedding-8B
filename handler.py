import time
import os
import torch
import runpod
from sentence_transformers import SentenceTransformer

# =====================================================
# Logging helper
# =====================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# =====================================================
# Model configuration
# =====================================================
MODEL_PATH = "/models/hf/qwen3-vl-embedding-8b"
model = None

# =====================================================
# Load Qwen3-VL-Embedding-8B Model
# =====================================================
def load_model():
    global model
    if model is not None:
        return

    log(f"Loading Qwen3-VL-Embedding-8B from {MODEL_PATH}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if local path exists; if not, fallback to HF Hub identifier for testing
    if not os.path.exists(MODEL_PATH):
        log(f"WARNING: Model path {MODEL_PATH} not found. Falling back to Hugging Face Hub (Qwen/Qwen3-VL-Embedding-8B)")
        model_name_or_path = "Qwen/Qwen3-VL-Embedding-8B"
    else:
        model_name_or_path = MODEL_PATH

    # Initialize SentenceTransformer with modern configurations
    model = SentenceTransformer(
        model_name_or_path,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32},
        device=device
    )
    
    # Enable CUDA optimizations if available
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
    log(f"Model successfully loaded on device: {device}")

# =====================================================
# RunPod handler
# =====================================================
def handler(event):
    log("Handler started")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"CUDA device: {torch.cuda.get_device_name(0)}")

    input_data = event["input"]
    
    # Extract inputs according to RAG request specification
    system_prompt = input_data.get("system_prompt")
    user_prompt = input_data.get("user_prompt")
    temperature = input_data.get("temperature")  # Logged but ignored as embeddings are deterministic
    
    if temperature is not None:
        log(f"Received temperature: {temperature} (ignored for deterministic embedding generation)")

    if not user_prompt:
        return {"error": "user_prompt is required"}

    # Ensure model is preloaded
    load_model()

    # Format the input using the instruction-aware pattern:
    # "Instruct: {task_description}\nQuery: {query}"
    if isinstance(user_prompt, list):
        if system_prompt:
            texts_to_embed = [f"Instruct: {system_prompt}\nQuery: {q}" for q in user_prompt]
        else:
            texts_to_embed = user_prompt
    else:
        if system_prompt:
            texts_to_embed = f"Instruct: {system_prompt}\nQuery: {user_prompt}"
        else:
            texts_to_embed = user_prompt

    log("Starting embedding generation...")
    start_time = time.time()
    
    # Generate embeddings
    embeddings = model.encode(texts_to_embed)
    
    log(f"Embedding generation completed in {time.time() - start_time:.4f}s")
    
    # Convert numpy output to list
    if hasattr(embeddings, "tolist"):
        embeddings_list = embeddings.tolist()
    else:
        embeddings_list = embeddings
        
    return {
        "embedding": embeddings_list
    }

# =====================================================
# Start RunPod serverless
# =====================================================
runpod.serverless.start({"handler": handler})