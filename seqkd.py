import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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


MODEL_ID = "gpt2"   # GPT-2 small, ~120M
TEACHER_MODEL_ID = "MiniLLM/teacher-gpt2-1.5B"

TRAIN_FILE = "dolly-processed/full/gpt2/train.jsonl"
VALID_FILE = "dolly-processed/full/gpt2/valid.jsonl"

BASE_OUTPUT_DIR = "outputs/gpt2_seqKD_V3_dolly"
BASE_BEST_MODEL_DIR = "outputs/gpt2_seqKD_V3_dolly_best"

MAX_LENGTH = 512
NUM_EPOCHS = 20
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 32
LEARNING_RATES = [5e-5]
WEIGHT_DECAY = 0.01
SEED = 42
TEST_SIZE = 500

ALPHA = 0.5
TEMPERATURE = 1.0

TRAIN_MAX_NEW_TOKENS = 64  # much cheaper than 300 for training rollout
EVAL_MAX_NEW_TOKENS = 300      # keep long generation only for evaluation
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


class CudaTimer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        torch.cuda.synchronize()
        self.start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()
        torch.cuda.synchronize()
        self.ms = self.start.elapsed_time(self.end)

from torch.profiler import profile, record_function, ProfilerActivity

def profile_a_few_steps(trainer, num_steps=5):
    model = trainer.model
    model.train()

    dataloader = trainer.get_train_dataloader()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            batch = trainer._prepare_inputs(batch)

            model.zero_grad(set_to_none=True)

            with record_function("train_step"):
                loss = trainer.compute_loss(model, batch)
                loss.backward()

            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

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
from dataclasses import dataclass

@dataclass
class LeftPadCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features):
        responses = None
        if "response" in features[0]:
            responses = [f["response"] for f in features]

        token_features = []
        for f in features:
            item = {
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
            }
            token_features.append(item)

        batch = self.tokenizer.pad(
            token_features,
            padding=True,              # dynamic padding to longest in batch
            return_tensors="pt",
        )

        if responses is not None:
            batch["response"] = responses

        return batch
    
def build_prompt_only_dataset(dataset, tokenizer, max_length: int):
    def tokenize_prompt_only(examples):
        enc = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    prompt_dataset = dataset.map(
        tokenize_prompt_only,
        batched=True,
        remove_columns=["prompt","response"],
    )
    
    print("Padding side inside build_prompt_only_dataset:", tokenizer.padding_side)
    print("\nPrompt-only generation dataset ready.")
    return prompt_dataset

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
            padding=False,
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
            max_new_tokens=300,
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




class seqKDTrainer(Trainer):
    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_dataset,
        teacher_model,
        eval_prompt_dataset,
        generation_max_new_tokens,
        alpha=0.5,
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
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        self.eval_prompt_dataset = eval_prompt_dataset
        self.generation_max_new_tokens = generation_max_new_tokens
        self.alpha = alpha
        self.temperature = temperature
        self.model_accepts_loss_kwargs = False


    @torch.no_grad()
    def generate_mixed_rollout_cached(
        self,
        teacher,
        student,
        input_ids,
        attention_mask,
        max_new_tokens,
        alpha,
        temperature,
        pad_token_id,
        eos_token_id,
    ):
        """
        Efficient rollout:
        - one full prefill on the prompt
        - then cached decoding with 1 token at a time
        - stores teacher log-probs for generated positions
        """
        device = input_ids.device
        B = input_ids.size(0)

        generated_ids = input_ids.clone()
        current_attention_mask = attention_mask.clone()
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        teacher_out = teacher(
            input_ids=generated_ids,
            attention_mask=current_attention_mask,
            return_dict=True,
            use_cache=True,
        )
    
        student_out = student(
            input_ids=generated_ids,
            attention_mask=current_attention_mask,
            return_dict=True,
            use_cache=True,
        )
    
        teacher_past = teacher_out.past_key_values
        student_past = student_out.past_key_values

        teacher_next_logits = teacher_out.logits[:, -1, :] / temperature
        student_next_logits = student_out.logits[:, -1, :] / temperature

        teacher_log_probs_steps = []
        gen_loss_mask_steps = []

        for _ in range(max_new_tokens):
            active = ~finished  # rows still generating
            teacher_log_probs = F.log_softmax(teacher_next_logits, dim=-1)
            teacher_probs = teacher_log_probs.exp()
            student_probs = F.softmax(student_next_logits, dim=-1)
            mixed_probs = alpha * teacher_probs + (1.0 - alpha) * student_probs
            mixed_probs = mixed_probs / mixed_probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(mixed_probs, num_samples=1)
            # For rows already finished, append PAD and do not count them in loss
            next_token = torch.where(
                active.unsqueeze(1),
                next_token,
                torch.full_like(next_token, pad_token_id),
            )

            # Save teacher distribution for this generated position
            teacher_log_probs_steps.append(teacher_log_probs)
            gen_loss_mask_steps.append(active.float())

            # Append token
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Append attention mask: 1 only for rows that were still active
            next_attn = active.to(current_attention_mask.dtype).unsqueeze(1)
            current_attention_mask = torch.cat([current_attention_mask, next_attn], dim=1)

            # Rows that just produced EOS/PAD become finished
            just_finished = (
                (next_token.squeeze(1) == eos_token_id) |
                (next_token.squeeze(1) == pad_token_id)
            )
            finished = finished | just_finished

            if finished.all():
                break
        
            teacher_out = teacher(
                input_ids=next_token,
                attention_mask=current_attention_mask,
                past_key_values=teacher_past,
                return_dict=True,
                use_cache=True,
            )
        

        
            student_out = student(
                input_ids=next_token,
                attention_mask=current_attention_mask,
                past_key_values=student_past,
                return_dict=True,
                use_cache=True,
            )
            
            teacher_past = teacher_out.past_key_values
            student_past = student_out.past_key_values

            teacher_next_logits = teacher_out.logits[:, -1, :] / temperature
            student_next_logits = student_out.logits[:, -1, :] / temperature

        if len(teacher_log_probs_steps) == 0:
            vocab = teacher.config.vocab_size
            teacher_log_probs_steps = torch.empty(B, 0, vocab, device=device)
            gen_loss_mask = torch.empty(B, 0, device=device)
        else:
            teacher_log_probs_steps = torch.stack(teacher_log_probs_steps, dim=1)  # [B, T, V]
            gen_loss_mask = torch.stack(gen_loss_mask_steps, dim=1)                # [B, T]

        return generated_ids, current_attention_mask, teacher_log_probs_steps, gen_loss_mask

    def compute_reverse_kl_loss_from_teacher_cache(
        self,
        student,
        generated_ids,
        attention_mask,
        prompt_padded_len,
        teacher_log_probs_steps,
        gen_loss_mask,
        temperature=1.0,
    ):
        """
        Uses cached teacher distributions from rollout.
        Only re-runs the student with gradients.
        """
        T = teacher_log_probs_steps.size(1)
        if T == 0:
            return generated_ids.new_zeros((), dtype=torch.float32)

        student_out = student(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        student_logits = student_out.logits[:, :-1, :] / temperature
        student_log_probs = F.log_softmax(student_logits, dim=-1)

        # Generated token positions begin at prompt_padded_len - 1 in logits
        start_idx = max(prompt_padded_len - 1, 0)
        student_log_probs_gen = student_log_probs[:, start_idx:start_idx + T, :]   # [B, T, V]
        student_probs_gen = student_log_probs_gen.exp()

        per_pos = (
            student_probs_gen * (teacher_log_probs_steps - student_log_probs_gen)
        ).sum(dim=-1)   # [B, T]

        loss = -(per_pos * gen_loss_mask).sum() / gen_loss_mask.sum().clamp_min(1.0)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_padded_len = input_ids.size(1)

        was_training = model.training
        model.eval()

        try:
            
            generated_ids, generated_attention_mask, teacher_log_probs_steps, gen_loss_mask = (
                self.generate_mixed_rollout_cached(
                    teacher=self.teacher_model,
                    student=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.generation_max_new_tokens,
                    alpha=self.alpha,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            )
        
            generated_ids = generated_ids.clone()
            generated_attention_mask = generated_attention_mask.clone()
            teacher_log_probs_steps = teacher_log_probs_steps.clone()
            gen_loss_mask = gen_loss_mask.clone()

        finally:
            if was_training:
                model.train()

                
        loss = self.compute_reverse_kl_loss_from_teacher_cache(
            student=model,
            generated_ids=generated_ids,
            attention_mask=generated_attention_mask,
            prompt_padded_len=prompt_padded_len,
            teacher_log_probs_steps=teacher_log_probs_steps,
            gen_loss_mask=gen_loss_mask,
            temperature=self.temperature,
        )
        

        if return_outputs:
            return loss, {
                "generated_ids": generated_ids,
                "generated_attention_mask": generated_attention_mask,
            }

        return loss



    @torch.no_grad()
    def evaluate_rouge_l(self, prompt_dataset, metric_key_prefix="eval"):
        self.model.eval()
        device = self.model.device

        dataloader = DataLoader(
            prompt_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )

        predictions = []
        references = []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
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
        metrics = super().evaluate(
            eval_dataset=eval_dataset if eval_dataset is not None else self.eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

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
    train_dataset,
    prompt_eval_dataset,
    training_args,
    tokenizer,
):
    data_collator = LeftPadCollator(tokenizer)
    
    trainer = seqKDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset["validation"],
        teacher_model=teacher_model,
        eval_prompt_dataset=prompt_eval_dataset,
        generation_max_new_tokens=TRAIN_MAX_NEW_TOKENS,   # not 300
        alpha=ALPHA,
        temperature=TEMPERATURE,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # profile_a_few_steps(trainer, num_steps=5)


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
    train_dataset,
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
        train_dataset=train_dataset,
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

    tokenizer.padding_side='left'
    train_dataset = build_prompt_only_dataset(formatted_dataset, tokenizer, MAX_LENGTH)

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
            train_dataset=train_dataset,
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