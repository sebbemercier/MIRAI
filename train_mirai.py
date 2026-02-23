# Copyright 2026 The OpenSLM Project
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.trainer import OpenSLMTrainer

def run():
    print("🔥 Training MIRAI (Sales Expert)...")
    trainer = OpenSLMTrainer(model_id="Qwen/Qwen2.5-1.5B-Instruct", output_dir="./mirai-fine-tuned")
    trainer.train(data_path="MIRAI/data/train_sales.jsonl", epochs=1, batch_size=1, grad_accum=16)

if __name__ == "__main__":
    run()
