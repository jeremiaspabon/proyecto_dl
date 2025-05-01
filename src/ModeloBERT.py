"""
Script: modelo_bert.py
Descripcion: Entrenamiento y fine-tuning de modelo BERT para clasificacion multiclase.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, get_scheduler, BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd

# =============================
# CONFIGURACION Y RUTAS
# =============================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

TOKENIZER_NAME = 'bert-base-multilingual-cased'
MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# CARGA DE DATOS
# ============================
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_data.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_data.csv'))

# ============================
# TOKENIZACION Y CODIFICACION
# ============================
def clean_and_validate_texts(texts):
    valid_texts = []
    for text in texts:
        if isinstance(text, str):
            valid_texts.append(text)
        elif text is None or pd.isna(text):
            valid_texts.append("")
        else:
            try:
                valid_texts.append(str(text))
            except:
                valid_texts.append("")
    return valid_texts

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

train_encodings = tokenizer(
    clean_and_validate_texts(train_df['Reporte'].tolist()),
    truncation=True,
    padding='max_length',
    max_length=MAX_LEN,
    return_tensors='pt'
)

test_encodings = tokenizer(
    clean_and_validate_texts(test_df['Reporte'].tolist()),
    truncation=True,
    padding='max_length',
    max_length=MAX_LEN,
    return_tensors='pt'
)

label_encoder = LabelEncoder()
train_labels_encoded = torch.tensor(label_encoder.fit_transform(train_df['Categoria']))
test_labels_encoded = torch.tensor(label_encoder.transform(test_df['Categoria']))
num_labels = len(label_encoder.classes_)

# ============================
# MANEJO DE DESBALANCE DE CLASES
# ============================
class_weights = torch.tensor([
    len(train_labels_encoded) / (num_labels * (train_labels_encoded == i).sum()) for i in range(num_labels)
], dtype=torch.float).to(DEVICE)

# ============================
# PREPARAR DATASETS Y DATALOADERS
# ============================
train_dataset = TensorDataset(
    train_encodings['input_ids'],
    train_encodings['attention_mask'],
    train_labels_encoded
)
test_dataset = TensorDataset(
    test_encodings['input_ids'],
    test_encodings['attention_mask'],
    test_labels_encoded
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ============================
# CONFIGURACION DEL MODELO BERT
# ============================
model = BertForSequenceClassification.from_pretrained(TOKENIZER_NAME, num_labels=num_labels)
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=NUM_EPOCHS * len(train_loader)
)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# ============================
# ENTRENAMIENTO CON EARLY STOPPING
# ============================
print("\n[INICIO] Entrenamiento del modelo BERT\n")

best_val_accuracy = 0
best_model_state = None
patience = 2
no_improve_epochs = 0
train_losses = []
val_accuracies = []

model.train()
for epoch in range(NUM_EPOCHS):
    print(f"\n[Epoch {epoch+1}/{NUM_EPOCHS}]\n" + "="*30)
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoca {epoch+1}/{NUM_EPOCHS}", leave=True)
    for batch in loop:
        input_ids, attention_mask, labels = [b.to(DEVICE) for b in batch]

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validacion
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(DEVICE) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = correct / total
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = model.state_dict().copy()
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    model.train()

if best_model_state:
    model.load_state_dict(best_model_state)
    print(f"Mejor modelo restaurado con Val Accuracy: {best_val_accuracy:.4f}")

# ============================
# EVALUACION FINAL
# ============================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [b.to(DEVICE) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nReporte de clasificación:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# ============================
# MATRIZ DE CONFUSION
# ============================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================
# GUARDAR MODELO Y TOKENS
# ============================
model_save_path = os.path.join(MODEL_DIR, 'modelo_bert_finetuned.pth')
torch.save(model.state_dict(), model_save_path)

with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

tokenizer.save_pretrained(os.path.join(MODEL_DIR, 'tokenizer'))

print("\n[FIN] Modelo BERT entrenado y guardado exitosamente.")
