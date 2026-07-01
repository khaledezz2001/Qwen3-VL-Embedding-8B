import time
import os
import runpod
import subprocess
import threading
import requests

# =====================================================
# Logging helper
# =====================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# =====================================================
# Configuration
# =====================================================
MODEL_PATH = "/models/hf/qwen3-8b"
VLLM_PORT = 8000
VLLM_HEALTH_URL = f"http://localhost:{VLLM_PORT}/health"
VLLM_COMPLETIONS_URL = f"http://localhost:{VLLM_PORT}/v1/chat/completions"

vllm_process = None

def stream_output(pipe, prefix):
    for line in pipe:
        print(f"[{prefix}] {line}", end="", flush=True)

def start_vllm_server():
    """Start the vLLM server as a background process."""
    global vllm_process

    log("Starting vLLM server...")
    model_name_or_path = MODEL_PATH if os.path.exists(MODEL_PATH) else "Qwen/Qwen3-8B"

    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name_or_path,
        "--port", str(VLLM_PORT),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--max-model-len", "32768",
        "--max-num-seqs", "8",
        "--gpu-memory-utilization", "0.90",
        "--logits_processors",
        "vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
    ]

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    vllm_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    # Stream vLLM output in a background thread
    t = threading.Thread(target=stream_output, args=(vllm_process.stdout, "OUT"), daemon=True)
    t.start()

    # Wait for the server to be ready
    max_wait = 300  # 5 minutes
    start = time.time()
    while time.time() - start < max_wait:
        try:
            r = requests.get(VLLM_HEALTH_URL, timeout=2)
            if r.status_code == 200:
                log(f"vLLM server ready in {time.time() - start:.1f}s")
                return True
        except requests.ConnectionError:
            pass

        # Check if process died
        if vllm_process.poll() is not None:
            log(f"vLLM server exited with code {vllm_process.returncode}")
            return False

        time.sleep(2)

    log("vLLM server timed out!")
    return False

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
    return_logprobs = input_data.get("logprobs", 0)
    
    if not user_prompt:
        log("Error: user_prompt is required")
        return {"error": "user_prompt is required"}

    log(f"Incoming Request - System Prompt: {system_prompt}")
    log(f"Incoming Request - User Prompt: {user_prompt}")

    log("Starting text generation...")
    start_time = time.time()
    
    is_list = isinstance(user_prompt, list)
    prompts_to_process = user_prompt if is_list else [user_prompt]
    
    response_data = []
    logprobs_data = []
    
    model_name_or_path = MODEL_PATH if os.path.exists(MODEL_PATH) else "Qwen/Qwen3-8B"

    for idx, q in enumerate(prompts_to_process):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": q})

        payload = {
            "model": model_name_or_path,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature if temperature and temperature > 0.0 else 0.0,
            "top_p": top_p if top_p else 1.0,
        }
        
        if return_logprobs > 0:
            payload["logprobs"] = True
            payload["top_logprobs"] = return_logprobs
            
        try:
            r = requests.post(VLLM_COMPLETIONS_URL, json=payload)
            r.raise_for_status()
            out = r.json()
            
            choice = out["choices"][0]
            text = choice["message"]["content"]
            response_data.append(text)
            
            if return_logprobs > 0:
                logprobs_data.append(choice.get("logprobs", {}))
                
            prompt_tokens = out["usage"]["prompt_tokens"]
            completion_tokens = out["usage"]["completion_tokens"]
            finish_reason = choice["finish_reason"]
            log(f"vLLM State (Req {idx}) - Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}, Finish Reason: {finish_reason}")
            
        except Exception as e:
            err_msg = f"API request failed: {str(e)}"
            if 'r' in locals() and r is not None and hasattr(r, 'text'):
                err_msg += f" Response: {r.text}"
            log(err_msg)
            return {"error": err_msg}

    log(f"Text generation completed in {time.time() - start_time:.4f}s")
    
    if not is_list:
        response_data = response_data[0]
        log(f"Generated Response: {response_data}")
        result = {"response": response_data}
        if return_logprobs > 0:
            result["logprobs"] = logprobs_data[0]
        return result
    else:
        log(f"Generated Responses: {response_data}")
        result = {"response": response_data}
        if return_logprobs > 0:
            result["logprobs"] = logprobs_data
        return result

# =====================================================
# Start RunPod serverless
# =====================================================
if start_vllm_server():
    runpod.serverless.start({"handler": handler})
else:
    log("Failed to start vLLM server. Exiting.")
    os._exit(1)
