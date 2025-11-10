Proyecto: Detección de Fraude en E‑commerce (PyTorch)

Resumen
-------
Clasificador binario para detectar transacciones potencialmente fraudulentas en un dataset de e‑commerce.
El flujo incluye: EDA básica, split estratificado (70/15/15), preprocesamiento mínimo, MLP en PyTorch con
BCEWithLogitsLoss y pos_weight para desbalanceo, y evaluación con métricas y gráficas adecuadas para clases desbalanceadas.


Estructura mínima del proyecto
------------------------------
```text
.
├─ 01_proyecto.ipynb         # Notebook principal
├─ transactions.csv          # Dataset en CSV
└─ README.md                 # Este archivo


Dataset
-------
Fuente: https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset

Cómo usarlo aquí:
1) Descarga el dataset desde Kaggle.
2) Exporta/coloca el CSV como 'transactions.csv' en la carpeta raíz del proyecto.
3) El notebook elimina la columna 'transaction_id' y crea la columna 'hour' desde 'transaction_time'.


Requisitos
----------
- Python 3.10+ recomendado
- Librerías principales: pandas, numpy, matplotlib, scikit-learn, torch, ipython


Ejecución
---------
1) Abre el notebook (Jupyter o VS Code) y ejecuta las celdas en orden.
2) El script realiza un split estratificado:
      - train: 70%
      - val  : 15%
      - test : 15%
3) Preprocesamiento (después del split, ajustado SOLO con train):
      - Numéricas continuas: z-score (media y std del train).
      - Binarias 0/1: se dejan tal cual (promo_used, avs_match, cvv_result, three_ds_flag).
      - Categóricas de texto: one-hot con columnas fijadas por el train (reindex en val/test).
4) Modelo:
      - MLP simple: 64 → 32 → 1 (Dropout 0.2).
      - Loss: BCEWithLogitsLoss(pos_weight=neg/pos) para compensar ~2% positivos.
      - Opt: Adam(lr=1e-3, weight_decay=1e-4).
5) Dataloaders:
      - batch_size=512, shuffle en train.
      - num_workers=4, pin_memory=True, persistent_workers=True (Linux/WSL). 
        Si hay problemas de multiproceso: reduce num_workers a 0–2 o desactiva persistent_workers.
6) Métricas y gráficas (test):
      - Umbral óptimo por F1 en validación.
      - Reporte en test @0.5 y @mejor umbral: accuracy, precision, recall, F1, AUROC, AUPRC y matriz de confusión.
      - Gráficas: ROC, Precision–Recall (con baseline por prevalencia), matriz de confusión y distribución de scores.


Columnas esperadas por el código
--------------------------------
Target:
  - is_fraud (0/1)

Numéricas (se escalan): 
  - account_age_days, total_transactions_user, avg_amount_user, amount, shipping_distance_km, hour

Binarias (se dejan 0/1):
  - promo_used, avs_match, cvv_result, three_ds_flag

Categóricas (one-hot):
  - country, bin_country, channel, merchant_category

Notas: El notebook construye 'hour' a partir de 'transaction_time' y luego elimina 'transaction_time'.


Detalles clave sobre el desbalanceo
-----------------------------------
- Se calcula pos_weight = (#negativos / #positivos) en el train y se pasa a BCEWithLogitsLoss.
- No se aplica sigmoid antes de la loss (la loss ya lo hace internamente).
- Para evaluar, se aplica sigmoid a los logits y se ajusta el umbral con validación (F1 u otro criterio).


Reproducibilidad
----------------
- Semillas: torch.manual_seed(42), random_state=42 en los splits.
- Los parámetros de estandarización y el espacio one‑hot se aprenden SOLO del train.


Rendimiento y recursos
----------------------
- Con TensorDataset en memoria, num_workers>0 aporta mejora moderada; en CPUs limitadas puede no ser necesario.
- Con GPU y pin_memory=True, usa non_blocking=True al mover tensores para acelerar las copias.


Solución de problemas
---------------------
- Error de tipos en BCEWithLogitsLoss: asegurar targets como float y misma forma que logits (N,1).
- Problemas con multiproceso en WSL/Windows: baja num_workers y desactiva persistent_workers.
- NaNs tras one-hot: al reindex de dummies usar fill_value=0 (el notebook lo aplica).
- Si aparece división por cero en el escalado: las std=0 se pueden sustituir por 1.0 (ya contemplado al reemplazar si es necesario).


Licencias y uso
---------------
- El dataset es propiedad de sus autores y está sujeto a la licencia indicada en Kaggle. Revísala antes de redistribuir.
- Este proyecto es educativo y no incluye garantías. Úsalo bajo tu propia responsabilidad.