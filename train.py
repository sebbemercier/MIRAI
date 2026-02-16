# Copyright 2026 The OpenSLM Project
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

def train_mirai(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Device détecté : {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "train_ecommerce_chat.jsonl")
    dataset = load_dataset("json", data_files=data_path, split="train")

    def tokenize_function(examples):
        prompts = [f"User: {u}\nAssistant: {r}" for u, r in zip(examples['instruction'], examples['response'])]
        return tokenizer(prompts, padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    args = TrainingArguments(
        output_dir="./mirai-output",
        per_device_train_batch_size=8,
        num_train_epochs=1,
        use_mps_device=True if device == "mps" else False,
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized_datasets)
    print("--- Démarrage de l'entraînement MIRAI ---")
    trainer.train()
    model.save_pretrained("./fine_tuned_mirai")
    print("✅ MIRAI entraîné et sauvegardé.")

if __name__ == "__main__":
    train_mirai()
