# ============================================================
# 1. Imports
# ============================================================
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)

from rouge_metric import compute_metrics


# ============================================================
# 2. Configuration
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL_ID = "gpt2"   # GPT-2 small, ~120M
TEACHER_MODEL_ID = "MiniLLM/teacher-gpt2-1.5B"

TRAIN_FILE = "dolly-processed/full/gpt2/train.jsonl"
VALID_FILE = "dolly-processed/full/gpt2/valid.jsonl"

BASE_OUTPUT_DIR = "outputs/gpt2_reverseKD_dolly"
BASE_BEST_MODEL_DIR = "outputs/gpt2_reverseKD_dolly_best_"

MAX_LENGTH = 512
NUM_EPOCHS = 20
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 32
LEARNING_RATES = [5e-5]
WEIGHT_DECAY = 0.01
SEED = 42
TEST_SIZE = 500

KD_RATIO = 0.5
TEMPERATURE = 1.0

EVAL_MAX_NEW_TOKENS = 300
device = "cuda" if torch.cuda.is_available() else "cpu"


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

    return dataset

# ============================================================
# 6. Format examples
# ============================================================
def format_example(example):
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    output_text = str(example.get("output", "")).strip()

    prompt = f"Instruction: {instruction}"
    if input_text:
        prompt += f" Input: {input_text}"
    prompt += " Response:"

    return {
        "prompt": prompt,
        "response": output_text,
    }


def preprocess_dataset(dataset):
    formatted_dataset = dataset.map(format_example)

    keep_cols = ["prompt", "response"]
    formatted_dataset = formatted_dataset.remove_columns(
        [col for col in formatted_dataset["train"].column_names if col not in keep_cols]
    )

    return formatted_dataset



# ============================================================
# 7. Load tokenizer
# ============================================================
def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nTokenizer loaded.")
    return tokenizer


# ============================================================
# 8. Tokenize dataset for KD training
#    - prompt + response
#    - labels only on response region
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
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    print("\nTokenized KD dataset ready.")
    print("Padding side inside dataset tokenizer:", tokenizer.padding_side)
    return tokenized_dataset


# ============================================================
# 9. Build prompt-only dataset for Rouge-L generation eval
#    Keeps "response" so we can compare generated outputs
# ============================================================
def build_prompt_generation_dataset(dataset, tokenizer, max_length: int):
    def tokenize_prompt_only(examples):
        enc = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "response": examples["response"],
        }

    prompt_dataset = dataset.map(
        tokenize_prompt_only,
        batched=True,
        remove_columns=["prompt"],
    )
    prompt_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
        output_all_columns=True,
    )
    print("Padding side inside build_prompt_generation_dataset:", tokenizer.padding_side)
    print("\nPrompt-only generation dataset ready.")
    return prompt_dataset

# ============================================================
# 10. Load models
# ============================================================
def load_models(student_model_id: str, tokenizer, teacher_model_id: str = None, device = device):
    print(f"\nLoading student from: {student_model_id}")
    student = AutoModelForCausalLM.from_pretrained(student_model_id)
    student.config.pad_token_id = tokenizer.eos_token_id
    student.to(device)
    student.train()

    teacher = None
    if teacher_model_id is not None:
        print(f"Loading teacher from: {teacher_model_id}")
        teacher = AutoModelForCausalLM.from_pretrained(teacher_model_id)
        teacher.config.pad_token_id = tokenizer.eos_token_id
        teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    print("\nModel(s) loaded successfully.")
    return student, teacher


# ============================================================
# 11. Training arguments
#    Uses Rouge-L as best-model metric
# ============================================================
def get_training_args(output_dir: str, learning_rate: float):
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_rougeL",
        greater_is_better=True,
        save_total_limit=20,
        logging_steps=50,
        logging_dir=f"runs/{os.path.basename(output_dir)}",
        logging_strategy="steps",
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
    )
# ============================================================
# 12. Save best model
# ============================================================
def save_best_model(trainer, tokenizer, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nBest model saved to: {save_dir}")


# ============================================================
# 13. Test generation
# ============================================================
def test_generation(model, tokenizer):
    prompt = """Instruction: Explain how to calculate an ADC map from diffusion MRI. Response:"""

    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nGenerated text:")
    print("=" * 80)
    print(generated_text)
    print("=" * 80)

# ============================================================
# 14. Plot Curves
# ============================================================

def plot_training_curves(trainer, save_dir=None):
    log_history = trainer.state.log_history

    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    rouge_steps, rouge_vals = [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("step"))
            train_losses.append(entry["loss"])

        if "eval_loss" in entry:
            eval_steps.append(entry.get("step"))
            eval_losses.append(entry["eval_loss"])

        if "eval_rougeL" in entry:
            rouge_steps.append(entry.get("step"))
            rouge_vals.append(entry["eval_rougeL"])

    plt.figure(figsize=(8, 5))
    if train_steps:
        plt.plot(train_steps, train_losses, label="Train loss")
    if eval_steps:
        plt.plot(eval_steps, eval_losses, marker="o", label="Validation loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        loss_path = os.path.join(save_dir, "loss_curve.png")
        plt.savefig(loss_path, dpi=200, bbox_inches="tight")
        print(f"Loss curve saved to: {loss_path}")
    plt.show()

    if rouge_steps:
        plt.figure(figsize=(8, 5))
        plt.plot(rouge_steps, rouge_vals, marker="o", label="Validation Rouge-L")
        plt.xlabel("Step")
        plt.ylabel("Rouge-L")
        plt.title("Validation Rouge-L")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir is not None:
            rouge_path = os.path.join(save_dir, "rougeL_curve.png")
            plt.savefig(rouge_path, dpi=200, bbox_inches="tight")
            print(f"Rouge-L curve saved to: {rouge_path}")
        plt.show()

# ============================================================
# 15. Custom KD Trainer with Rouge-L evaluation
# ============================================================
class KDTrainer(Trainer):
    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_dataset,
        teacher_model,
        eval_prompt_dataset,
        generation_max_new_tokens,
        kd_ratio=0.5,
        temperature=1.0,
        tokenizer=None,
        data_collator=None,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        self.teacher_model = teacher_model
        self.eval_prompt_dataset = eval_prompt_dataset
        self.generation_max_new_tokens = generation_max_new_tokens
        self.kd_ratio = kd_ratio
        self.temperature = temperature
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]

        # Student forward
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            return_dict=True,
        )
        student_logits = outputs.logits

        # Standard LM loss on gold response tokens
        shift_student_logits = student_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        lm_loss = F.cross_entropy(
            shift_student_logits.view(-1, shift_student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Teacher forward
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=attention_mask,
                return_dict=True,
            )
            teacher_logits = teacher_outputs.logits

        shift_teacher_logits = teacher_logits[:, :-1, :].contiguous()
        valid_mask = (shift_labels != -100)

        # Reverse KD: KL(student || teacher)
        teacher_log_probs = F.log_softmax(
            shift_teacher_logits / self.temperature,
            dim=-1,
        )
        student_log_probs = F.log_softmax(
            shift_student_logits / self.temperature,
            dim=-1,
        )
        student_probs = student_log_probs.exp()

        kd_per_token = (
            student_probs * (student_log_probs - teacher_log_probs)
        ).sum(dim=-1)

        kd_loss = kd_per_token[valid_mask].mean()
        kd_loss = kd_loss * (self.temperature ** 2)

        loss = (1.0 - self.kd_ratio) * lm_loss + self.kd_ratio * kd_loss

        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def evaluate_rouge_l(self, prompt_dataset, metric_key_prefix="eval"):
        self.model.eval()
        device = self.model.device

        dataloader = DataLoader(
            prompt_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
        )

        predictions = []
        references = []

        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.generation_max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            generated_ids = outputs[:, input_ids.size(1):]

            pred_responses = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )

            predictions.extend([pred.strip() for pred in pred_responses])
            references.extend([[ref] for ref in batch["response"]])

        metrics = compute_metrics(predictions, references, xlingual=False)
        return {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # First get eval_loss on the tokenized eval set
        metrics = super().evaluate(
            eval_dataset=eval_dataset if eval_dataset is not None else self.eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Then add generation-based Rouge-L on the prompt-only eval set
        rouge_metrics = self.evaluate_rouge_l(
            prompt_dataset=self.eval_prompt_dataset,
            metric_key_prefix=metric_key_prefix,
        )

        metrics.update(rouge_metrics)
        self.log(rouge_metrics)
        return metrics


# ============================================================
# 16. Train
# ============================================================
def train_model(
    model,
    teacher_model,
    tokenized_dataset,
    prompt_eval_dataset,
    training_args,
    tokenizer,
):
    
    trainer = KDTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        teacher_model=teacher_model,
        eval_prompt_dataset=prompt_eval_dataset,
        generation_max_new_tokens=EVAL_MAX_NEW_TOKENS,
        kd_ratio=KD_RATIO,
        temperature=TEMPERATURE,
        tokenizer=tokenizer,
    )

    train_result = trainer.train()

    print("\nTraining finished.")
    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best validation Rouge-L:", trainer.state.best_metric)

    return trainer, train_result


# ============================================================
# 17. Evaluate on test set with Rouge-L
# ============================================================
def evaluate_on_test(trainer, prompt_test_dataset):
    test_metrics = trainer.evaluate_rouge_l(
        prompt_dataset=prompt_test_dataset,
        metric_key_prefix="test",
    )

    print("\nTest set results:")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    return test_metrics

# ============================================================
# 18. One experiment
# ============================================================
def run_experiment(
    learning_rate,
    tokenized_dataset,
    prompt_eval_dataset,
    prompt_test_dataset,
    tokenizer,
):
    lr_name = f"{learning_rate:.0e}"
    lr_name = lr_name.replace("e-0", "e-")

    output_dir = f"{BASE_OUTPUT_DIR}_lr_{lr_name}"
    best_model_dir = f"{BASE_BEST_MODEL_DIR}_lr_{lr_name}"

    print("\n" + "#" * 80)
    print(f"Starting training for learning rate = {learning_rate}")
    print(f"Output dir: {output_dir}")
    print(f"Best model dir: {best_model_dir}")
    print("#" * 80)

    model, teacher_model = load_models(
        student_model_id=MODEL_ID,
        tokenizer=tokenizer,
        teacher_model_id=TEACHER_MODEL_ID, device = device
    )

    training_args = get_training_args(output_dir, learning_rate)

    trainer, _ = train_model(
        model=model,
        teacher_model=teacher_model,
        tokenized_dataset=tokenized_dataset,
        prompt_eval_dataset=prompt_eval_dataset,
        training_args=training_args,
        tokenizer=tokenizer,
    )
    save_best_model(trainer, tokenizer, best_model_dir)
    plot_training_curves(trainer, save_dir=output_dir)

    test_metrics = evaluate_on_test(trainer, prompt_test_dataset)
    
    test_generation(trainer.model, tokenizer)

    return {
        "learning_rate": learning_rate,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_val_rougeL": trainer.state.best_metric,
        "test_metrics": test_metrics,
        "best_model_dir": best_model_dir,
    }


# ============================================================
# 18. Main
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

    tokenizer.padding_side='right'
    tokenized_dataset = tokenize_dataset(formatted_dataset, tokenizer, MAX_LENGTH)

    tokenizer.padding_side = "left"
    prompt_eval_dataset = build_prompt_generation_dataset(
        formatted_dataset["validation"],
        tokenizer,
        MAX_LENGTH,
    )
    prompt_test_dataset = build_prompt_generation_dataset(
        formatted_dataset["test"],
        tokenizer,
        MAX_LENGTH,
    )
    
    all_results = []

    for lr in LEARNING_RATES:
        seed_everything(SEED)
        result = run_experiment(
            learning_rate=lr,
            tokenized_dataset=tokenized_dataset,
            prompt_eval_dataset=prompt_eval_dataset,
            prompt_test_dataset=prompt_test_dataset,
            tokenizer=tokenizer,
        )
        all_results.append(result)

    print("\n" + "=" * 80)
    print("Summary of all learning-rate runs:")
    for r in all_results:
        print(f"LR: {r['learning_rate']}")
        print(f"  Best checkpoint: {r['best_checkpoint']}")
        print(f"  Best validation Rouge-L: {r['best_val_rougeL']}")
        print(f"  Saved best model to: {r['best_model_dir']}")
        print(f"  Test metrics: {r['test_metrics']}")
        print("-" * 80)


if __name__ == "__main__":
    main()