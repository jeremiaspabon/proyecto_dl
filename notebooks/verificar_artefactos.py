from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path
import torch
import pickle

print("\n📂 Iniciando verificación de artefactos...")

# Definir rutas
ROOT_DIR = Path(__file__).parent.parent.resolve()
MODEL_DIR = ROOT_DIR / "models"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"
MODEL_FILE = MODEL_DIR / "modelo_bert_finetuned.pth"
ENCODER_FILE = MODEL_DIR / "label_encoder.pkl"

# Verificar tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)
    tokens = tokenizer.tokenize("Ejemplo de texto para prueba.")
    print("✅ Tokenizer cargado correctamente.")
    print("🧾 Tokens de ejemplo:", tokens)
except Exception as e:
    print("❌ Error al cargar el tokenizer:", e)

# Verificar modelo (usando state_dict)
try:
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=4)
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    model.eval()
    print("✅ Modelo cargado correctamente (state_dict).")
except Exception as e:
    print("❌ Error al cargar el modelo:", e)

# Verificar label encoder
try:
    with open(ENCODER_FILE, "rb") as f:
        label_encoder = pickle.load(f)
    print("✅ Label encoder cargado correctamente.")
    print("🏷️ Clases:", list(label_encoder.classes_))
except Exception as e:
    print("❌ Error al cargar label encoder:", e)
