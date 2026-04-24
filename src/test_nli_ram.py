from transformers import pipeline
import psutil, os
b = psutil.Process(os.getpid()).memory_info().rss / 1024**2
nli = pipeline('zero-shot-classification', model='cross-encoder/nli-MiniLM2-L6-H768')
a = psutil.Process(os.getpid()).memory_info().rss / 1024**2
print(f'NLI RAM: {a-b:.0f} MB')
