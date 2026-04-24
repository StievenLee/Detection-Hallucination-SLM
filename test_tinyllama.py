from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu"  # aman untuk laptop
)

prompt = """<|system|>
You are a helpful assistant.
<|user|>
Explain artificial intelligence in simple terms.
<|assistant|>
"""
# prompt = "Explain artificial intelligence in simple terms."

inputs = tokenizer(prompt, return_tensors="pt")

M = 5

responses = []

for i in range(M):
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    response = outputs[0][inputs["input_ids"].shape[-1]:]

    text = tokenizer.decode(response, skip_special_tokens=True)
    responses.append(text)

    print(f"\n=== Sample {i+1} ===")
    print(text)