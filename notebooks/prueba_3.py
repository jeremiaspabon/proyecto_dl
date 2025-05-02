import torch

checkpoint = torch.load("models/modelo_bert_finetuned.pth", map_location="cpu")

print("Tipo:", type(checkpoint))
if isinstance(checkpoint, dict):
    print("Primeras claves:", list(checkpoint.keys())[:5])
else:
    print("⚠️ Este archivo es un modelo completo, no un state_dict")
