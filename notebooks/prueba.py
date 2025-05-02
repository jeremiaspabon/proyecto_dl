from transformers import BertTokenizer
from pathlib import Path

# Ruta absoluta al tokenizer desde el archivo actual (ubicado en notebooks/)
tokenizer_path = (Path(__file__).parent.parent / "models" / "tokenizer").resolve(strict=True)

# Cargar tokenizer desde ruta absoluta local
tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)

# Probar tokenización
texto = "Este es un ejemplo de prueba para revisar el vocabulario."
tokens = tokenizer.tokenize(texto)

print("✅ Tokenizer cargado correctamente.")
print("📦 Tokens generados:", tokens)
