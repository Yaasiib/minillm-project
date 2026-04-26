import os
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# =========================
# Config
# =========================
CSV_PATH = "selfinst_eval_results.csv"
OUT_PATH = "selfinst_eval_results_with_gpt4_scores.csv"

JUDGE_MODEL = "gpt-4.1"   # or "gpt-4o"
MAX_ROWS = 20           # set to 20 for a cheap test run
SLEEP_SEC = 0.2           # small pause between requests

# =========================
# Helpers
# =========================
def parse_json_safely(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        raise

def build_judge_prompt(instruction: str, input_text: str, reference: str, model_answer: str):
    return f"""You are an expert evaluator of instruction-following quality.

Please score TWO responses to the same instruction:
1) the ground-truth reference answer
2) the model-generated answer

Use a 1-10 scale for each response.
Judge correctness, helpfulness, completeness, and faithfulness to the instruction.

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

def judge_one(client: OpenAI, model: str, instruction: str, input_text: str, reference: str, model_answer: str):
    prompt = build_judge_prompt(instruction, input_text, reference, model_answer)

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    return parse_json_safely(response.output_text)

# =========================
# Main
# =========================
def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=api_key)

    df = pd.read_csv(CSV_PATH)

    required = ["instruction", "reference", "sft_response", "distilled_response"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # allow either input_text or input
    if "input_text" not in df.columns:
        if "input" in df.columns:
            df["input_text"] = df["input"]
        else:
            df["input_text"] = ""

    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS].copy()

    ref_total_for_sft = 0.0
    sft_total = 0.0

    ref_total_for_dist = 0.0
    distilled_total = 0.0

    sft_ref_scores = []
    sft_model_scores = []
    dist_ref_scores = []
    dist_model_scores = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Judging"):
        instruction = str(row["instruction"])
        input_text = str(row["input_text"])
        reference = str(row["reference"])
        sft_response = str(row["sft_response"])
        distilled_response = str(row["distilled_response"])

        # Judge SFT answer
        sft_json = judge_one(
            client=client,
            model=JUDGE_MODEL,
            instruction=instruction,
            input_text=input_text,
            reference=reference,
            model_answer=sft_response,
        )

        time.sleep(SLEEP_SEC)

        # Judge distilled answer
        dist_json = judge_one(
            client=client,
            model=JUDGE_MODEL,
            instruction=instruction,
            input_text=input_text,
            reference=reference,
            model_answer=distilled_response,
        )

        time.sleep(SLEEP_SEC)

        sft_ref = float(sft_json["reference_score"])
        sft_model = float(sft_json["model_score"])
        dist_ref = float(dist_json["reference_score"])
        dist_model = float(dist_json["model_score"])

        sft_ref_scores.append(sft_ref)
        sft_model_scores.append(sft_model)
        dist_ref_scores.append(dist_ref)
        dist_model_scores.append(dist_model)

        ref_total_for_sft += sft_ref
        sft_total += sft_model

        ref_total_for_dist += dist_ref
        distilled_total += dist_model

    df["sft_reference_score"] = sft_ref_scores
    df["sft_model_score"] = sft_model_scores
    df["dist_reference_score"] = dist_ref_scores
    df["dist_model_score"] = dist_model_scores

    sft_ratio = sft_total / ref_total_for_sft if ref_total_for_sft > 0 else None
    distilled_ratio = distilled_total / ref_total_for_dist if ref_total_for_dist > 0 else None

    print("\n================ GPT JUDGE RESULTS ================")
    print(f"SFT GPT ratio       : {sft_ratio:.4f}")
    print(f"Distilled GPT ratio : {distilled_ratio:.4f}")

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved judged CSV to: {OUT_PATH}")

if __name__ == "__main__":
    main()