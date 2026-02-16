# Copyright 2026 The OpenSLM Project
# Licensed under the Apache License, Version 2.0
import sentencepiece as spm
from ATLAS.model import AtlasAgent
# On importera MUSE plus tard pour le contenu SEO

class MiraiSalesAgent:
    def __init__(self, tokenizer_path="models/ecommerce_tokenizer.model"):
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self.atlas = AtlasAgent() # MIRAI utilise ATLAS pour la connaissance produit

    def process_query(self, user_text):
        print(f"\n--- MIRAI Processing: '{user_text}' ---")
        
        # 1. Analyse des tokens pour détecter l'intention
        ids = self.sp.encode_as_ids(user_text)
        pieces = self.sp.encode_as_pieces(user_text)
        
        # Logique simplifiée d'intention (sera remplacée par un petit modèle de classification)
        intent = self._classify_intent(user_text.lower())
        
        if intent == "PRODUCT_QUERY":
            # Extraire le SKU (pour le test on simule l'extraction)
            sku = self._extract_sku(user_text)
            print(f"MIRAI: Intention détectée -> Information Produit pour {sku}")
            
            # Appel à ATLAS
            product_info = self.atlas.handle_query(sku)
            return self._format_sales_response(product_info)
            
        elif intent == "GREETING":
            return "Bonjour ! Je suis l'assistant de vente OpenSLM. Comment puis-je vous aider aujourd'hui ?"
            
        else:
            return "Je ne suis pas sûr de comprendre. Voulez-vous des informations sur un produit ou vérifier un stock ?"

    def _classify_intent(self, text):
        if any(word in text for word in ["stock", "disponible", "poids", "matière", "taille"]):
            return "PRODUCT_QUERY"
        if any(word in text for word in ["bonjour", "hello", "hi"]):
            return "GREETING"
        return "UNKNOWN"

    def _extract_sku(self, text):
        # Simulation d'extraction de SKU via NLP
        # Dans un vrai SLM, ce serait un modèle de NER (Named Entity Recognition)
        if "nike" in text: return "NIKE-123"
        return "GENERIC-ITEM"

    def _format_sales_response(self, atlas_data):
        # Ici on pourrait appeler MUSE pour rendre le texte plus "vendeur"
        response = f"Absolument ! Voici ce que j'ai trouvé :\n\n{atlas_data}\n"
        response += "\nSouhaitez-vous que je l'ajoute à votre panier ?"
        return response

if __name__ == "__main__":
    mirai = MiraiSalesAgent()
    # Test d'un flux complet
    print(mirai.process_query("Bonjour, est-ce que les Nike sont en stock ?"))
