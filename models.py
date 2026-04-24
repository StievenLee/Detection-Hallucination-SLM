from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",
    "Qwen/Qwen1.5-1.8B-Chat"
]

prompt = """<|system|>
You are a helpful assistant.
<|user|>
Explain artificial intelligence in simple terms.
<|assistant|>
"""

M = 3  # cukup kecil dulu buat test

for model_name in models:
    print(f"\n==============================")
    print(f"Loading {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    start_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu"
    )
    print(f"Load time: {time.time() - start_load:.2f}s")

    inputs = tokenizer(prompt, return_tensors="pt")

    responses = []
    start_gen = time.time()

    for i in range(M):
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

        response = outputs[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(response, skip_special_tokens=True)
        responses.append(text)

        print(f"\n--- Sample {i+1} ---")
        print(text)

    print(f"Generation time: {time.time() - start_gen:.2f}s")