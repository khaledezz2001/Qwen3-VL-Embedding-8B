import time
import os
import torch
import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer

# =====================================================
# Logging helper
# =====================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# =====================================================
# Model configuration
# =====================================================
MODEL_PATH = "/models/hf/qwen3-8b-instruct"
model = None
tokenizer = None

# =====================================================
# Load Qwen3-8B-Instruct Model and Tokenizer
# =====================================================
def load_model():
    global model, tokenizer
    if model is not None and tokenizer is not None:
        return

    log(f"Loading Qwen3-8B-Instruct from {MODEL_PATH}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if local path exists; if not, fallback to HF Hub identifier for testing
    if not os.path.exists(MODEL_PATH):
        log(f"WARNING: Model path {MODEL_PATH} not found. Falling back to Hugging Face Hub (Qwen/Qwen3-8B-Instruct)")
        model_name_or_path = "Qwen/Qwen3-8B-Instruct"
    else:
        model_name_or_path = MODEL_PATH

    # Initialize Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Enable CUDA optimizations if available
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
    log(f"Model and tokenizer successfully loaded on device: {device}")

# =====================================================
# Text generation helper
# =====================================================
def generate_response(system_prompt, user_prompt, gen_kwargs):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            **gen_kwargs
        )
    
    input_len = model_inputs.input_ids.shape[1]
    output_ids = generated_ids[0][input_len:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)

# =====================================================
# RunPod handler
# =====================================================
def handler(event):
    log("Handler started")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"CUDA device: {torch.cuda.get_device_name(0)}")

    input_data = event["input"]
    
    # Extract inputs
    system_prompt = input_data.get("system_prompt")
    user_prompt = input_data.get("user_prompt")
    temperature = input_data.get("temperature", 0.7)
    top_p = input_data.get("top_p", 0.9)
    max_new_tokens = input_data.get("max_new_tokens", 1024)
    
    if not user_prompt:
        return {"error": "user_prompt is required"}

    # Ensure model is loaded
    load_model()

    # Determine generation parameters
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
    }
    if temperature and temperature > 0.0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        if top_p:
            gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    log("Starting text generation...")
    start_time = time.time()
    
    if isinstance(user_prompt, list):
        responses = [generate_response(system_prompt, q, gen_kwargs) for q in user_prompt]
        response_data = responses
    else:
        response_data = generate_response(system_prompt, user_prompt, gen_kwargs)
        
    log(f"Text generation completed in {time.time() - start_time:.4f}s")
    
    return {
        "response": response_data
    }

# =====================================================
# Start RunPod serverless
# =====================================================
runpod.serverless.start({"handler": handler})
