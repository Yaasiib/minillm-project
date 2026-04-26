# ============================================================
# Evaluate SFT vs Distilled model on SelfInst
# Metrics:
#   1) ROUGE-L
#   2) GPT-4 judge ratio (optional)
# ============================================================

import os
import json
import glob
from dataclasses import dataclass

import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
import evaluate

from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional GPT judge
USE_GPT4_JUDGE = False
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# 1. Config
# ============================================================

@dataclass
class Config:
    sft_model_path: str = "outputs/gpt2-medium-dolly-best"
    distilled_model_path: str = "outputs/minillm_v1_student_best"
    selfinst_file: str = "data/self-inst/valid.jsonl"

    max_prompt_length: int = 256
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False

    output_csv: str = "selfinst_eval_results.csv"
    judge_model: str = "gpt-4.1"


CFG = Config()


# ============================================================
# 2. Device
# ============================================================

def get_device():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


# ============================================================
# 3. Load SelfInst
# ============================================================

def find_selfinst_files(selfinst_dir):
    jsonl_files = glob.glob(os.path.join(selfinst_dir, "**", "*.jsonl"), recursive=True)
    json_files = glob.glob(os.path.join(selfinst_dir, "**", "*.json"), recursive=True)
    all_files = sorted(jsonl_files + json_files)

    if not all_files:
        raise FileNotFoundError(f"No json/jsonl files found under: {selfinst_dir}")

    print("Found files:")
    for f in all_files[:10]:
        print(" ", f)

    return all_files


from datasets import load_dataset

def load_selfinst_dataset(selfinst_file):
    ds = load_dataset("json", data_files={"eval": selfinst_file})["eval"]
    print(ds)
    print("Sample keys:", ds[0].keys())
    print("First example:", ds[0])
    return ds

def pick_field(example, candidates):
    for c in candidates:
        if c in example and example[c] is not None:
            return str(example[c]).strip()
    return ""


def build_prompt_and_reference(example):
    """
    Robust field handling for SelfInst-style files.
    """
    instruction = pick_field(example, ["instruction", "prompt", "question"])
    input_text = pick_field(example, ["input", "context"])
    reference = pick_field(example, ["output", "response", "answer", "target"])

    prompt = f"### Instruction:\n{instruction}\n\n"
    if input_text:
        prompt += f"### Input:\n{input_text}\n\n"
    prompt += "### Response:\n"

    return {
        "prompt": prompt,
        "reference": reference,
    }


# ============================================================
# 4. Load models
# ============================================================

def load_model_and_tokenizer(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    return tokenizer, model


# ============================================================
# 5. Generation
# ============================================================

@torch.no_grad()
def generate_response(model, tokenizer, prompt, cfg, device):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_prompt_length,
    ).to(device)

    out = model.generate(
        **enc,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    full_text = tokenizer.decode(out[0], skip_special_tokens=True)

    # keep only generated continuation after prompt if possible
    if full_text.startswith(prompt):
        response = full_text[len(prompt):].strip()
    else:
        response = full_text.strip()

    return response


# ============================================================
# 6. ROUGE-L
# ============================================================

def compute_rouge_l(predictions, references):
    rouge = evaluate.load("rouge")
    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    return result["rougeL"]


# ============================================================
# 7. GPT-4 judge prompt
# ============================================================

def build_judge_prompt(instruction, input_text, reference, model_answer):
    return f"""You are an expert evaluator of instruction-following quality.

Please score TWO responses to the same instruction:
1) the ground-truth reference answer
2) the model-generated answer

Use a 1-10 scale for each response.
Judge helpfulness, correctness, completeness, and faithfulness to the instruction.

Return STRICT JSON with these keys only:
{{
  "reference_score": <number>,
  "model_score": <number>
}}

Instruction:
{instruction}

Input:
{input_text}

Reference answer:
{reference}

Model answer:
{model_answer}
"""


def parse_json_safely(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # try to recover JSON substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        raise


# ============================================================
# 8. Optional GPT-4 judging
# ============================================================

def judge_with_gpt4(rows, cfg):
    if OpenAI is None:
        raise ImportError("openai package is not installed.")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    ref_total = 0.0
    sft_total = 0.0
    distilled_total = 0.0

    sft_scores = []
    distilled_scores = []
    ref_scores = []

    for row in tqdm(rows, desc="GPT-4 judging"):
        instruction = row["instruction"]
        input_text = row["input_text"]
        reference = row["reference"]

        # Judge SFT
        prompt_sft = build_judge_prompt(
            instruction=instruction,
            input_text=input_text,
            reference=reference,
            model_answer=row["sft_response"],
        )
        resp_sft = client.responses.create(
            model=cfg.judge_model,
            input=prompt_sft,
        )
        sft_json = parse_json_safely(resp_sft.output_text)

        # Judge distilled
        prompt_dist = build_judge_prompt(
            instruction=instruction,
            input_text=input_text,
            reference=reference,
            model_answer=row["distilled_response"],
        )
        resp_dist = client.responses.create(
            model=cfg.judge_model,
            input=prompt_dist,
        )
        dist_json = parse_json_safely(resp_dist.output_text)

        # We use the reference score from the same prompt call;
        # to stay consistent, average both reference scores.
        ref_score = (float(sft_json["reference_score"]) + float(dist_json["reference_score"])) / 2.0
        sft_score = float(sft_json["model_score"])
        distilled_score = float(dist_json["model_score"])

        ref_scores.append(ref_score)
        sft_scores.append(sft_score)
        distilled_scores.append(distilled_score)

        ref_total += ref_score
        sft_total += sft_score
        distilled_total += distilled_score

    sft_ratio = sft_total / ref_total if ref_total > 0 else None
    distilled_ratio = distilled_total / ref_total if ref_total > 0 else None

    return {
        "reference_scores": ref_scores,
        "sft_scores": sft_scores,
        "distilled_scores": distilled_scores,
        "sft_ratio": sft_ratio,
        "distilled_ratio": distilled_ratio,
    }


# ============================================================
# 9. Main evaluation
# ============================================================

def main():
    device = get_device()

    # Load dataset
    ds = load_selfinst_dataset(CFG.selfinst_file)

    # Build evaluation rows
    rows = []
    for ex in ds:
        rows.append({
            "topic": ex.get("topic", ""),
            "instruction": ex.get("instruction", ""),
            "input_text": ex.get("input", ""),
            "prompt": ex.get("prompt", ""),
            "reference": ex.get("output", ""),
        })

    print(f"Loaded {len(rows)} evaluation samples")

    # Load models
    print("\nLoading SFT model...")
    sft_tokenizer, sft_model = load_model_and_tokenizer(CFG.sft_model_path, device)

    print("\nLoading distilled model...")
    dist_tokenizer, dist_model = load_model_and_tokenizer(CFG.distilled_model_path, device)

    # Generate
    sft_predictions = []
    distilled_predictions = []
    references = []

    for row in tqdm(rows, desc="Generating"):
        prompt = row["prompt"]
        reference = row["reference"]

        sft_resp = generate_response(sft_model, sft_tokenizer, prompt, CFG, device)
        dist_resp = generate_response(dist_model, dist_tokenizer, prompt, CFG, device)

        row["sft_response"] = sft_resp
        row["distilled_response"] = dist_resp

        sft_predictions.append(sft_resp)
        distilled_predictions.append(dist_resp)
        references.append(reference)

    # ROUGE-L
    sft_rouge_l = compute_rouge_l(sft_predictions, references)
    distilled_rouge_l = compute_rouge_l(distilled_predictions, references)

    print("\n================ RESULTS ================")
    print(f"SFT ROUGE-L       : {sft_rouge_l:.4f}")
    print(f"Distilled ROUGE-L : {distilled_rouge_l:.4f}")

    # Optional GPT-4 judging
    if USE_GPT4_JUDGE:
        judge_result = judge_with_gpt4(rows, CFG)

        for i, row in enumerate(rows):
            row["reference_score"] = judge_result["reference_scores"][i]
            row["sft_score"] = judge_result["sft_scores"][i]
            row["distilled_score"] = judge_result["distilled_scores"][i]

        print(f"SFT GPT-4 ratio       : {judge_result['sft_ratio']:.4f}")
        print(f"Distilled GPT-4 ratio : {judge_result['distilled_ratio']:.4f}")

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(CFG.output_csv, index=False)
    print(f"\nSaved detailed outputs to: {CFG.output_csv}")


if __name__ == "__main__":
    main()