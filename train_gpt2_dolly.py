# ============================================================
# 1. Imports
# ============================================================
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)


# ============================================================
# 2. Configuration
# ============================================================
MODEL_ID = "gpt2"   # GPT-2 small, ~120M

TRAIN_FILE = "dolly-processed/full/gpt2/train.jsonl"
VALID_FILE = "dolly-processed/full/gpt2/valid.jsonl"

BASE_OUTPUT_DIR = "outputs/gpt2_small_dolly_SFT"
BASE_BEST_MODEL_DIR = "outputs/gpt2_small_dolly_best_SFT"

MAX_LENGTH = 512
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATES = [5e-5]
WEIGHT_DECAY = 0.01
SEED = 42
TEST_SIZE = 500   # fixed held-out test set


# ============================================================
# 3. Reproducibility
# ============================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# ============================================================
# 4. Check device
# ============================================================
def get_device():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("Using CPU")
    return "cpu"


# ============================================================
# 5. Load dataset
# ============================================================
def load_dolly_dataset(train_file: str, valid_file: str, test_size=500, seed=42):
    data_files = {
        "train": train_file,
        "validation": valid_file,
    }

    dataset = load_dataset("json", data_files=data_files)

    split_dataset = dataset["train"].train_test_split(
        test_size=test_size,
        seed=seed,
    )

    dataset["train"] = split_dataset["train"]
    dataset["test"] = split_dataset["test"]

    print(dataset)
    print("\nSample training example:")
    print(dataset["train"][0])
    print(f"\nTrain size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")
    print(f"Test size: {len(dataset['test'])}")

    return dataset


# ============================================================
# 6. Format examples
# ============================================================
def format_example(example):
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    output_text = str(example.get("output", "")).strip()

    prompt = f"### Instruction:\n{instruction}\n\n"
    if input_text:
        prompt += f"### Input:\n{input_text}\n\n"
    prompt += "### Response:\n"

    return {
        "prompt": prompt,
        "response": output_text,
    }


def preprocess_dataset(dataset):
    formatted_dataset = dataset.map(format_example)

    formatted_dataset = formatted_dataset.remove_columns(
        [col for col in formatted_dataset["train"].column_names if col not in ["prompt", "response"]]
    )

    print("\nFormatted sample:")
    print("PROMPT:\n", formatted_dataset["train"][0]["prompt"])
    print("RESPONSE:\n", formatted_dataset["train"][0]["response"])
    return formatted_dataset


# ============================================================
# 7. Load tokenizer
# ============================================================
def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nTokenizer loaded.")
    print("Pad token:", tokenizer.pad_token)
    print("EOS token:", tokenizer.eos_token)
    return tokenizer


# ============================================================
# 8. Tokenize dataset
# ============================================================
def tokenize_dataset(dataset, tokenizer, max_length: int):
    def tokenize_function(examples):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for prompt, response in zip(examples["prompt"], examples["response"]):
            full_text = prompt + response

            full_enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

            prompt_enc = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                padding=False,
            )

            input_ids = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]

            prompt_len = min(len(prompt_enc["input_ids"]), max_length)

            labels = input_ids.copy()
            labels[:prompt_len] = [-100] * prompt_len

            labels = [
                token if mask == 1 else -100
                for token, mask in zip(labels, attention_mask)
            ]

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "response"],
    )

    print("\nTokenized dataset ready.")
    return tokenized_dataset


# ============================================================
# 9. Load model
# ============================================================
def load_model(model_id: str, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.config.pad_token_id = tokenizer.eos_token_id

    print("\nModel loaded successfully.")
    return model


# ============================================================
# 10. Training arguments
# ============================================================
def get_training_args(output_dir: str, learning_rate: float):
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=50,
        logging_dir=f"runs/{os.path.basename(output_dir)}",
        logging_strategy="steps",
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
    )


# ============================================================
# 11. Train
# ============================================================
def train_model(model, tokenized_dataset, training_args):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    train_result = trainer.train()

    print("\nTraining finished.")
    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best validation loss:", trainer.state.best_metric)

    return trainer, train_result


# ============================================================
# 12. Evaluate on test set
# ============================================================
def evaluate_on_test(trainer, tokenized_dataset):
    test_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    print("\nTest set results:")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")
    return test_metrics


# ============================================================
# 13. Save best model
# ============================================================
def save_best_model(trainer, tokenizer, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nBest model saved to: {save_dir}")


# ============================================================
# 14. Test generation
# ============================================================
def test_generation(model, tokenizer):
    prompt = """### Instruction:
Explain how to calculate an ADC map from diffusion MRI.

### Response:
"""

    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nGenerated text:")
    print("=" * 80)
    print(generated_text)
    print("=" * 80)


# ============================================================
# 15. One experiment
# ============================================================
def run_experiment(learning_rate, tokenized_dataset, tokenizer):
    lr_name = f"{learning_rate:.0e}"   # e.g. 5e-04, 1e-04, 5e-05
    lr_name = lr_name.replace("e-0", "e-")  # make prettier: 5e-4

    output_dir = f"{BASE_OUTPUT_DIR}_lr_{lr_name}"
    best_model_dir = f"{BASE_BEST_MODEL_DIR}_lr_{lr_name}"

    print("\n" + "#" * 80)
    print(f"Starting training for learning rate = {learning_rate}")
    print(f"Output dir: {output_dir}")
    print(f"Best model dir: {best_model_dir}")
    print("#" * 80)

    model = load_model(MODEL_ID, tokenizer)
    training_args = get_training_args(output_dir, learning_rate)

    trainer, _ = train_model(
        model=model,
        tokenized_dataset=tokenized_dataset,
        training_args=training_args,
    )

    test_metrics = evaluate_on_test(trainer, tokenized_dataset)
    save_best_model(trainer, tokenizer, best_model_dir)
    test_generation(trainer.model, tokenizer)

    return {
        "learning_rate": learning_rate,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_val_loss": trainer.state.best_metric,
        "test_metrics": test_metrics,
        "best_model_dir": best_model_dir,
    }


# ============================================================
# 16. Main
# ============================================================
def main():
    seed_everything(SEED)
    get_device()

    dataset = load_dolly_dataset(
        TRAIN_FILE,
        VALID_FILE,
        test_size=TEST_SIZE,
        seed=SEED,
    )
    formatted_dataset = preprocess_dataset(dataset)

    tokenizer = load_tokenizer(MODEL_ID)
    tokenized_dataset = tokenize_dataset(formatted_dataset, tokenizer, MAX_LENGTH)

    all_results = []

    for lr in LEARNING_RATES:
        seed_everything(SEED)  # keep runs reproducible
        result = run_experiment(lr, tokenized_dataset, tokenizer)
        all_results.append(result)

    print("\n" + "=" * 80)
    print("Summary of all learning-rate runs:")
    for r in all_results:
        print(f"LR: {r['learning_rate']}")
        print(f"  Best checkpoint: {r['best_checkpoint']}")
        print(f"  Best val loss: {r['best_val_loss']}")
        print(f"  Saved best model to: {r['best_model_dir']}")
        print(f"  Test metrics: {r['test_metrics']}")
        print("-" * 80)


if __name__ == "__main__":
    main()