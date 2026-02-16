# Copyright 2026 The OpenSLM Project
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

def train_mirai(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    print(f"Fine-tuning MIRAI pour la vente...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("json", data_files="data/train_ecommerce_chat.jsonl", split="train")

    def tokenize_function(examples):
        # Formatage : Instruction (Intention) -> Réponse (Vente)
        prompts = [f"User: {u}
Assistant: {r}" for u, r in zip(examples['instruction'], examples['response'])]
        return tokenizer(prompts, padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    args = TrainingArguments(
        output_dir="./mirai-output",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=1e-4,
        save_strategy="epoch"
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized_datasets)
    trainer.train()
    model.save_pretrained("./fine_tuned_mirai")

if __name__ == "__main__":
    print("Script d'entraînement MIRAI prêt.")
