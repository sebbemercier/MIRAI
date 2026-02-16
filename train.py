# Copyright 2026 The OpenSLM Project
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import os

def train_mirai(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Device : {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "train_ecommerce_chat.jsonl")
    dataset = load_dataset("json", data_files=data_path, split="train")

    def tokenize_function(examples):
        prompts = [f"User: {u}\nAssistant: {r}</s>" for u, r in zip(examples['instruction'], examples['response'])]
        outputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=512)
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable()

    args = TrainingArguments(
        output_dir="./mirai-output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        use_mps_device=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized_datasets)
    trainer.train()
    model.save_pretrained("./fine_tuned_mirai")

if __name__ == "__main__":
    train_mirai()
