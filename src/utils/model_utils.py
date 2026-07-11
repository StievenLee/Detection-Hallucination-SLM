import gc
import time
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .metrics import ResourceMonitor, get_ram_usage_mb, get_model_size_mb


def set_seed(seed: int = 42):
    """
    Set semua sumber keacakan agar hasil reproducible.
    Panggil SEKALI di awal main() sebelum eksperimen.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # warn_only=True: jangan crash kalau ada op yang belum punya versi deterministik
    torch.use_deterministic_algorithms(True, warn_only=True)


# Chat template fallback manual untuk model yang tidak support apply_chat_template
PROMPT_TEMPLATES = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": (
        "<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
    ),
    "microsoft/phi-1_5": (
        "Human: {user}\nAI:"
    ),
    "Qwen/Qwen1.5-1.8B-Chat": (
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
}


def build_prompt(tokenizer, model_name: str, system: str, user: str) -> str:
    """
    Gunakan apply_chat_template jika tersedia,
    fallback ke template manual jika tidak.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    # Coba pakai built-in chat template
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            print(f"[Warning] apply_chat_template gagal: {e}. Pakai fallback.")

    # Fallback ke template manual
    template = PROMPT_TEMPLATES.get(model_name)
    if template:
        # Phi-2 tidak pakai system prompt
        if "{system}" in template:
            return template.format(system=system, user=user)
        else:
            return template.format(user=user)

    # Last resort: plain prompt
    print(f"[Warning] Tidak ada template untuk {model_name}, pakai plain prompt.")
    return f"System: {system}\n\nUser: {user}\n\nAssistant:"


def load_model_and_tokenizer(model_name: str, device: str = "cpu"):
    """Load tokenizer & model dengan monitoring RAM."""
    monitor = ResourceMonitor()
    print(f"\n{'='*50}")
    print(f"Loading: {model_name}")
    print(f"RAM sebelum load: {monitor.baseline_mb:.1f} MB")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Pastikan pad_token ada
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    start = time.time()
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     device_map=device,
    #     trust_remote_code=True,
    #     dtype=torch.float32,  # CPU-safe
    # )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    load_time = time.time() - start

    # Ukur dari parameter model langsung, bukan delta RAM proses
    model_size_mb = get_model_size_mb(model)
    snap = monitor.snapshot("after_load")

    print(f"Load time      : {load_time:.2f}s")
    print(f"Model size     : {model_size_mb:.1f} MB (dari parameter)")
    print(f"RAM tersisa    : {snap['ram_available_mb']:.1f} MB")

    return model, tokenizer, {
        "load_time_s": round(load_time, 3),
        "model_size_mb": round(model_size_mb, 1),
        "ram_after_load_mb": round(snap["ram_used_mb"], 1),
        "ram_available_mb": round(snap["ram_available_mb"], 1),
    }

def unload_model(model):
    """Bebaskan RAM setelah selesai pakai model."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[Cleanup] Model di-unload. RAM sekarang: {get_ram_usage_mb():.1f} MB")


def generate_responses(
    model,
    tokenizer,
    prompt: str,
    M: int = 10,
    max_new_tokens: int = 100,
    temperature: float = 0.5,
    top_p: float = 0.95,
    seed: int = None,
) -> tuple[list[str], dict]:
    """
    Generate M sampel dari satu prompt.
    Return: (list of response strings, timing & token stats)

    seed: kalau diisi, sampling M respons jadi reproducible antar-run
          (tetap acak ANTAR sampel dalam satu panggilan, tapi identik
          bila dijalankan ulang dengan seed sama).
    """
    # inputs = tokenizer(prompt, return_tensors="pt")
    # input_len = inputs["input_ids"].shape[-1]
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[-1]
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    responses = []
    total_new_tokens = 0
    start = time.time()

    for i in range(M):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                max_length=None,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][input_len:]
        total_new_tokens += len(new_tokens)
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        responses.append(text)

        print(f"  [Sample {i+1}/{M}] {len(new_tokens)} tokens → {text[:80]}...")

    gen_time = time.time() - start

    stats = {
        "gen_time_s": round(gen_time, 3),
        "total_new_tokens": total_new_tokens,
        "tokens_per_sec": round(total_new_tokens / gen_time, 2),
        "avg_tokens_per_sample": round(total_new_tokens / M, 1),
        "avg_time_per_sample_s": round(gen_time / M, 3),
    }

    print(f"  Gen time    : {gen_time:.2f}s")
    print(f"  Throughput  : {stats['tokens_per_sec']} tokens/s")

    return responses, stats


def generate_best_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 0.1,
    top_p: float = 0.95,
    seed: int = None,
) -> str:
    """
    Hasilkan SATU jawaban 'best generation' pada temperature rendah (default 0.1),
    mengikuti Farquhar et al. [3]. Dipakai HANYA untuk menilai correctness,
    TERPISAH dari M sampel T=0.5 yang dipakai menghitung semantic entropy.

    Temperature rendah -> jawaban paling mungkin (mode distribusi) -> stabil,
    tidak berisik seperti mengambil satu sampel acak dari T=0.5.

    seed: kalau diisi, satu jawaban ini reproducible antar-run (penting agar
          label 'correct' stabil, karena do_sample=True tetap dipertahankan).
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length=None,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()