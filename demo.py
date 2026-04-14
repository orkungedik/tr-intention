import torch
import torch.nn as nn
import json
import os
import string
from huggingface_hub import hf_hub_download

REPO_ID = "orkungedik/tr-intention"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TRintention(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, n_heads, n_layers):
        super(TRintention, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*4, 
            batch_first=True, dropout=0.1, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2)
        )

    def forward(self, input_ids):
        b = input_ids.shape[0]
        
        pad_mask = (input_ids == 0)
        cls_mask = torch.zeros((b, 1), dtype=torch.bool, device=input_ids.device)
        full_mask = torch.cat((cls_mask, pad_mask), dim=1)
        
        x = self.embed(input_ids)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x, src_key_padding_mask=full_mask)
        
        return self.classifier(x[:, 0])

def load_demo():
    print(f"Modeller {REPO_ID} adresinden indiriliyor...")
    
    # Hugging Face'den config ve model dosyalarını çek
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.bin")

    with open(config_path, 'r') as f:
        config = json.load(f)

    model = TRintention(
        vocab_size=config['vocab_size'],
        embed_dim=128, # Eğitimdeki değerler
        max_len=config['max_len'],
        n_heads=8,
        n_layers=4
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    return model, config

def run_inference(text, model, config):
    char_vocab = config['char_vocab']
    char_to_id = {char: i + 1 for i, char in enumerate(char_vocab)}
    max_len = config['max_len']

    ids = [char_to_id.get(c, 0) for c in text[:max_len]]
    padding = [0] * (max_len - len(ids))
    input_tensor = torch.tensor([ids + padding], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor) # input_tensor artık maske için forward'da doğrudan kullanılıyor
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
    label = "Fonksiyon" if pred_idx == 1 else "Soru"
    conf = probs[0][pred_idx].item()
    return label, conf

if __name__ == "__main__":
    model, config = load_demo()
    label, score = run_inference("bu durum sana ne ifade ediyor", model, config)
    print(f"Tahmin: {label} | Güven: {score:.2%}\n")
