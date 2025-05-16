from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from verl.utils.dataset.rl_dataset import collate_fn, tokenizer_wrapper
from verl import DataProto
from openai import OpenAI
import os
import time
import random


def external_batch_inference(client, requests, sampling_params, text_only=True, progress=False, external_api=False):
    params = sampling_params
    
    if external_api:
        model = os.environ.get("EXTERNAL_MODEL_NAME", "gpt-4o")
        print(model)
        if model == "gpt-4o" or model == "gpt-4o-mini":
            active_client = OpenAI()
        else:
            import json
            if os.environ.get("NVIDIA_API_KEY") is not None:
                nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
            else:
                nvidia_api_key = json.load(open("keys/nvidia_key.json", "r"))["key"]
            active_client = OpenAI(
                base_url = "https://integrate.api.nvidia.com/v1",
                api_key = nvidia_api_key
                )
        n_threads = 8
    else:
        active_client = client
        model = active_client.models.list().data[0].id
        n_threads = 256


    def get_completion(request, max_retries=5, backoff_base=5.0):
        assert isinstance(request, list) and all(isinstance(turn, dict) for turn in request), \
            "Format error. Should be a list of dictionaries"

        for attempt in range(max_retries):
            try:
                x = active_client.chat.completions.create(
                    model=model,
                    messages=request,
                    **params
                )
                return x
            except Exception as e:
                if hasattr(e, "status_code") and e.status_code in [429, 500, 502, 503, 504]:
                    wait_time = backoff_base * (1.5 ** attempt) + random.uniform(0, 1) # wait time will increase exponentially
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Non-retriable error: {e}")
        raise RuntimeError(f"Failed after {max_retries} retries.")

    with ThreadPoolExecutor(max_workers=min(len(requests), n_threads)) as executor:
        if progress:
            results = list(tqdm(
                executor.map(get_completion, requests),
                total=len(requests),
                desc=f"Inference (Parallel, Model: {model})" # Updated description
            ))
        else:
            results = list(executor.map(get_completion, requests))
    if text_only:
        results = [result.choices[0].message.content for result in results]
    
    if external_api:
        print("External API Inference Finished")
    else:
        print("Local Inference Finished")
    return results