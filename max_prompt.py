from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

dataset = load_dataset("MiniLLM/dolly-processed", split="train")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_prompts(batch):
    prompts = batch["prompt"]   # replace with your prompt field
    enc = tokenizer(prompts, add_special_tokens=False, truncation=False)
    return {"prompt_len": [len(x) for x in enc["input_ids"]]}

dataset = dataset.map(tokenize_prompts, batched=True, batch_size=1000)

lengths = dataset["prompt_len"]

print("Average:", np.mean(lengths))
print("Median:", np.median(lengths))
print("Std:", np.std(lengths))
print("95th percentile:", np.percentile(lengths, 95))
print("Max:", np.max(lengths))