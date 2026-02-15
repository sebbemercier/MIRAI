# Copyright 2026 The OpenSLM Project
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

import sentencepiece as spm

class MiraiSales:
    def __init__(self, tokenizer_path="../ATLAS/ecommerce_tokenizer.model"):
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

    def process_customer_query(self, text):
        # Tokenisation de la demande client
        tokens = self.sp.encode_as_pieces(text)
        print(f"MIRAI (Analyse): {tokens}")
        
        if "stock" in text.lower() or "disponible" in text.lower():
            return "CHECK_STOCK"
        return "GENERAL_QUERY"

if __name__ == "__main__":
    mirai = MiraiSales()
    action = mirai.process_customer_query("Bonjour, avez-vous du stock sur les Nike Air ?")
    print(f"Action déterminée : {action}")