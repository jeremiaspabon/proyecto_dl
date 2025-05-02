# proyecto_dl
Proyectos del curso deep learning para la categorización de PQR de una ONG
# Proyecto de Clasificación de Textos con BERT

Este proyecto tiene como objetivo desarrollar y validar un modelo de clasificación de texto basado en BERT multilingüe, aplicado a categorías como retroalimentación, solicitudes y reportes de usuarios.

## 🔍 Resultados de Evaluación

- **Accuracy general**: 85.4%
- **F1-Score macro**: 73.77%

### 📊 Métricas por clase

| Clase                     | Precisión | Recall | F1-Score |
|--------------------------|-----------|--------|----------|
| Insatisfacción mínima    | 0.52      | 0.72   | 0.60     |
| Retroalimentación positiva | 0.94     | 0.91   | 0.93     |
| Solicitud de asistencia  | 0.49      | 0.69   | 0.57     |
| Solicitud de información | 0.89      | 0.82   | 0.85     |

## 🖼️ Gráficas

- [`grafico_f1_score.png`](outputs/grafico_f1_score.png)
- [`grafico_precision_recall.png`](outputs/grafico_precision_recall.png)
- [`grafico_errores.png`](outputs/grafico_errores.png)
- [`confusion_matrix.png`](outputs/confusion_matrix.png)

## 🧪 Pruebas de Clasificación Manual

| Texto                                                       | Clasificación Esperada         | Clasificación Predicha           |
|-------------------------------------------------------------|--------------------------------|----------------------------------|
| Quiero saber más sobre los servicios disponibles.           | Solicitud de información       | Solicitud de información ✅       |
| Gracias por la excelente atención brindada.                 | Retroalimentación positiva     | Retroalimentación positiva ✅     |
| El sistema falló y no pude completar mi solicitud.          | Solicitud de asistencia        | Solicitud de información ❌       |
| Estoy inconforme con el trato recibido.                     | Insatisfacción mínima          | Retroalimentación positiva ❌     |

## 🛠️ Dificultades Encontradas

- ❗ Error al cargar `tokenizer` debido a rutas no compatibles con Hugging Face (`'models\\tokenizer'`).
- ❗ El modelo `.pth` entrenado contenía una clave inesperada (`bert.embeddings.position_ids`) incompatible con `from_pretrained`.
- ✅ Se corrigió el error eliminando dicha clave y guardando el estado con `strict=False`.
- ❗ Error de `pickle` con `label_encoder` por versiones incompatibles de NumPy (`ModuleNotFoundError: No module named 'numpy._core'`).
- ✅ Solucionado actualizando las dependencias desde `requirements.txt`.
- ❌ Intento fallido de subir `modelo_bert_finetuned.pth` a GitHub por exceder los 100MB.
- ✅ Se eliminó del historial de commits y se ignoró en `.gitignore`.

---
