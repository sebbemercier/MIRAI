import torch
import torch.nn as nn

class MiraiSLM(nn.Module):
    """
    SLM dédié à la conversion et stratégie de vente.
    Architecture: Multi-Layer Perceptron avec Gating Units.
    """
    def __init__(self, input_dim=256, hidden_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(), # Gating pour filtrer les infos pertinentes
            nn.Linear(hidden_dim, 128) # Vecteur de stratégie
        )
        self.upsell_head = nn.Linear(128, 10) # Top 10 produits suggérés

    def forward(self, features):
        latent = self.network(features)
        return self.upsell_head(latent)
