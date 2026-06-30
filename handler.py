import time
import os
import runpod
from vllm import LLM, SamplingParams

# =====================================================
# Logging helper
# =====================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# =====================================================
# Model configuration
# =====================================================
MODEL_PATH = "/models/hf/qwen3-8b"
llm = None

# =====================================================
# Load Qwen3-8B Model
# =====================================================
def load_model():
    global llm
    if llm is not None:
        return

    log(f"Loading Qwen3-8B from {MODEL_PATH} via vLLM")
    
    # Check if local path exists; if not, fallback to HF Hub identifier for testing
    if not os.path.exists(MODEL_PATH):
        log(f"WARNING: Model path {MODEL_PATH} not found. Falling back to Hugging Face Hub (Qwen/Qwen3-8B)")
        model_name_or_path = "Qwen/Qwen3-8B"
    else:
        model_name_or_path = MODEL_PATH

    # Initialize vLLM engine
    llm = LLM(
        model=model_name_or_path,
        trust_remote_code=True,
    )
        
    log(f"Model successfully loaded via vLLM")

# =====================================================
# RunPod handler
# =====================================================
def handler(event):
    log("Handler started")

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

    # Determine generation parameters for vLLM
    sampling_params = SamplingParams(
        temperature=temperature if temperature and temperature > 0.0 else 0.0,
        top_p=top_p if top_p else 1.0,
        max_tokens=max_new_tokens
    )

    log("Starting text generation...")
    start_time = time.time()
    
    tokenizer = llm.get_tokenizer()
    
    is_list = isinstance(user_prompt, list)
    prompts_to_process = user_prompt if is_list else [user_prompt]
    
    formatted_prompts = []
    for q in prompts_to_process:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": q})
        formatted_prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    # Generate using vLLM
    outputs = llm.generate(formatted_prompts, sampling_params=sampling_params, use_tqdm=False)
    
    if is_list:
        response_data = [out.outputs[0].text for out in outputs]
    else:
        response_data = outputs[0].outputs[0].text
        
    log(f"Text generation completed in {time.time() - start_time:.4f}s")
    
    return {
        "response": response_data
    }

# =====================================================
# Start RunPod serverless
# =====================================================
runpod.serverless.start({"handler": handler})
