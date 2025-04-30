"""
Script: preprocesamiento_bert.py
Descripcion: Preprocesamiento completo para clasificacion de textos con BERT.
             Incluye carga de datos, limpieza, normalizacion, analisis exploratorio,
             particion 70/30, tokenizacion con BERT y exportacion de tensores.
Autor: Integrante 1 - Proyecto de Maestria en Analitica de Datos
"""

import os
import re
import json
import torch
import pickle
import unicodedata
import requests
import pandas as pd
from io import BytesIO
from datetime import datetime
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# ============================
# CONFIGURACION Y RUTAS
# ============================
GITHUB_URL = "https://github.com/Carot2/MaestriaAnaliticaDatos/raw/c181d78bb214cd719fe04386fc90628b6e7922e4/dataset_ong2.xlsx"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)
TOKENIZER_NAME = 'bert-base-multilingual-cased'
MAX_LEN = 128

# ============================
# FUNCIONES DE LIMPIEZA
# ============================
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    text = re.sub(r'[^\w\s.,;:?!-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_date(col):
    try:
        col = pd.to_datetime(col, errors='coerce')
        return col.dt.strftime('%d/%m/%Y')
    except:
        return col

# ============================
# CARGA DE DATOS
# ============================
def load_data_from_github(url=GITHUB_URL):
    try:
        r = requests.get(url)
        r.raise_for_status()
        if url.endswith('.xlsx'):
            df = pd.read_excel(BytesIO(r.content))
        else:
            df = pd.read_csv(BytesIO(r.content))
        print(f"Datos cargados: {df.shape}")
        return df
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return pd.DataFrame()

# ============================
# ANALISIS EXPLORATORIO
# ============================
def run_eda(df):
    eda = {
        'total_registros': len(df),
        'columnas': list(df.columns),
        'valores_nulos': df.isnull().sum().to_dict(),
        'tipos': df.dtypes.astype(str).to_dict(),
        'categorias_unicas': df['Categoria'].nunique(),
        'distribucion_categorias': df['Categoria'].value_counts().to_dict(),
        'longitudes_texto': {
            'media': df['Reporte'].str.len().mean(),
            'mediana': df['Reporte'].str.len().median(),
            'min': df['Reporte'].str.len().min(),
            'max': df['Reporte'].str.len().max()
        }
    }
    path = os.path.join(OUTPUT_DIR, 'data_analysis.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(eda, f, indent=2, ensure_ascii=False)
    print(f"EDA guardado en: {path}")
    return eda

# ============================
# LIMPIEZA Y NORMALIZACION
# ============================
def clean_data(df):
    df = df.copy()
    df['Reporte'] = df['Reporte'].apply(normalize_text)
    df['Resolucion'] = df['Resolucion'].apply(normalize_text)
    df['Municipio'] = df['Municipio'].apply(normalize_text)
    df['Fecha'] = normalize_date(df['Fecha'])
    df.dropna(subset=['Reporte', 'Categoria'], inplace=True)
    df.drop_duplicates(subset=['Reporte', 'Categoria'], inplace=True)
    return df

# ============================
# PARTICION Y GUARDADO
# ============================
def split_and_save(df):
    X = df['Reporte']
    y = df['Categoria']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    pd.DataFrame({'Reporte': X_train, 'Categoria': y_train}) \
        .to_csv(os.path.join(OUTPUT_DIR, 'train_data.csv'), index=False)
    pd.DataFrame({'Reporte': X_test, 'Categoria': y_test}) \
        .to_csv(os.path.join(OUTPUT_DIR, 'test_data.csv'), index=False)

    print("Conjuntos de entrenamiento y prueba guardados.")
    return X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()

# ============================
# METADATOS DEL CONJUNTO
# ============================
def save_metadata(df):
    meta = {
        'fecha_procesamiento': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'columnas': list(df.columns),
        'categorias': df['Categoria'].unique().tolist(),
        'distribucion': df['Categoria'].value_counts().to_dict(),
        'tamaño_total': len(df),
        'descripcion': 'Dataset limpio y particionado para modelo BERT'
    }
    path = os.path.join(OUTPUT_DIR, 'dataset_metadata.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Metadatos guardados en: {path}")

# ============================
# TOKENIZACION CON BERT
# ============================
def tokenize_and_save(train_texts, test_texts, train_labels, test_labels):
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

    train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
    test_encodings = tokenizer(test_texts, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')

    data_dict = {
        'train_encodings': {
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask']
        },
        'test_encodings': {
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask']
        },
        'train_labels': train_labels,
        'test_labels': test_labels
    }

    out_path = os.path.join(OUTPUT_DIR, 'bert_tokenized_data.pt')
    torch.save(data_dict, out_path)
    print(f"Datos tokenizados guardados en: {out_path}")

# ============================
# MAIN: PIPELINE COMPLETO
# ============================
if __name__ == "__main__":
    print("\n[INICIO] Preprocesamiento de datos para BERT")

    df = load_data_from_github()

    if df.empty:
        print("\n[ERROR] No se pudo continuar con el procesamiento.")
    else:
        df = clean_data(df)
        run_eda(df)
        save_metadata(df)
        X_train, X_test, y_train, y_test = split_and_save(df)
        tokenize_and_save(X_train, X_test, y_train, y_test)

    print("\n[FIN] Proceso de preprocesamiento completado.\n")
