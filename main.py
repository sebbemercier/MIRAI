from fastapi import FastAPI
import httpx
import torch
import asyncio
from transformers import pipeline

app = FastAPI(title="MIRAI | Le Stratège")

ATLAS_URL = "http://localhost:8001"
IRIS_URL = "http://localhost:8002"

print("Chargement du SLM MIRAI (TinyLlama)...")
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

@app.get("/sales-strategy/{product_id}")
async def get_strategy(product_id: int):
    async with httpx.AsyncClient() as client:
        # MIRAI interroge ATLAS et IRIS
        p_res = await client.get(f"{ATLAS_URL}/product/{product_id}")
        a_res = await client.get(f"{IRIS_URL}/analyze/{product_id}")
        
        product = p_res.json()
        analysis = a_res.json()

    prompt = f"<|system|>\nTu es MIRAI, un expert en conversion de vente. Propose une stratégie promotionnelle unique basée sur l'analyse suivante.</s>\n<|user|>\nProduit: {product['name']}\nAnalyse de IRIS: {analysis['ai_analysis']}</s>\n<|assistant|>\n"
    
    outputs = pipe(prompt, max_new_tokens=60, do_sample=True, temperature=0.8)
    strategy = outputs[0]["generated_text"].split("<|assistant|>\n")[-1].strip()
    
    return {
        "target_product": product["name"],
        "strategy": strategy
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
