# Copyright 2026 The OpenSLM Project
# Licensed under the Apache License, Version 2.0

import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Ajout du chemin racine pour permettre les imports entre dossiers (ATLAS, MUSE, etc.)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MIRAI.main import MiraiSalesAgent

app = FastAPI(title="OpenSLM API Gateway", version="1.0.0")

# Configuration CORS pour le Frontend (React/Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En prod, remplace par l'URL de ton front
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instance de l'agent MIRAI
sales_agent = MiraiSalesAgent()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

@app.get("/")
def read_root():
    return {"message": "OpenSLM API is running", "engine": "MIRAI"}

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Point d'entrée principal pour le Frontend.
    Prend un message, l'analyse via MIRAI, interroge ATLAS et rédige via MUSE.
    """
    try:
        if not request.message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Le flux complet : Intent -> Data -> SEO Copy
        result = sales_agent.process_query(request.message)
        
        return ChatResponse(response=result)
        
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Lancement du serveur sur le port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
