# 💳 Scoring de Riesgo Crediticio y Workstation de Originación (ML + SHAP)

Sistema *end-to-end* de evaluación de riesgo crediticio construido sobre el dataset **Give Me Some Credit** (150.000 solicitudes reales), integrando auditoría relacional en **SQL (DuckDB)**, ingeniería de características financieras libre de *data leakage*, modelado comparativo con **XGBoost / LightGBM** y despliegue interactivo en **Streamlit Cloud** con explicabilidad individual mediante **SHAP**.

🚀 **Aplicación Interactiva en Vivo:** https://scoring-riesgo-crediticio-ml.streamlit.app/

---

## 🎯 Problema de Negocio y Enfoque Metodológico

En originación crediticia, el objetivo no es maximizar la exactitud global (*Accuracy*), sino discriminar con precisión qué solicitantes tienen alta probabilidad de incurrir en mora grave (**90 días o más de atraso en los próximos 2 años**, clase minoritaria con prevalencia del **6.68%**) y definir un **punto de corte financiero** que equilibre el margen por buenos pagadores frente al costo de los préstamos incobrables.

### Flujo de Ingeniería
1. **Ingesta Relacional y Auditoría SQL (`sql/01_analisis_exploratorio.sql`):** Carga en DuckDB y análisis de cohortes que detectó el efecto escalonado del historial de mora (desde `2.73%` sin atrasos hasta `41.21%` con mora de 90+ días) y un segmento crítico de códigos centinela (`96`/`98`) con `54.65%` de incumplimiento.
2. **Preprocesamiento Modular Sin *Data Leakage* (`src/preprocessing.py`):** Transformador compatible con `scikit-learn` (`BaseEstimator`, `TransformerMixin`) que aprende medianas y percentiles de acotamiento (*winsorization*) exclusivamente sobre `Train` y construye ratios financieros (`indice_severidad_atrasos`, `gasto_mensual_estimado`, `ingreso_per_capita`).
3. **Validación Cruzada Estratificada y Optimización Económica (`src/metrics.py`):** Evaluación mediante `StratifiedKFold (k=5)` y desplazamiento del umbral de decisión ($0.50 \rightarrow 0.170$) para maximizar el beneficio neto esperado.
4. **Explicabilidad Regulatoria (`notbook/03_interpretabilidad_shap.ipynb`):** Auditoría global y local con `shap.TreeExplainer`, validando que `RevolvingUtilization_clean` (`0.7319`) e `indice_severidad_atrasos` (`0.4169`) lideran el poder predictivo[cite: 5].

---

## 📊 Resultados en Conjunto de Evaluación (Test Set — 30.000 registros)

| Configuración | Umbral | ROC-AUC | PR-AUC | Estadístico KS | Recall (Default) | F2-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Corte Estándar)** | 0.500 | 0.8697 | 0.4084 | 0.5844 | 19.50% | 0.2255 |
| **XGBoost (Corte Financiero Óptimo)** | **0.170** | **0.8697** | **0.4084** | **0.5844** | **56.36%** | **0.5097** |[cite: 4]

---

## 🗂️ Estructura del Proyecto

```text
Scoring-Riesgo-Crediticio-ML/
├── .streamlit/
│   └── config.toml                          # Tema visual Fintech Dark
├── data/
│   └── processed/
│       └── lote_prueba_100.csv              # Lote sintético con outliers y nulos para Batch Scoring
├── models/
│   └── modelo_crediticio.joblib             # Pipeline entrenado + umbral financiero óptimo
├── notbook/
│   ├── 01_eda_y_sql.ipynb                   # Auditoría relacional con DuckDB y análisis de hipótesis
│   ├── 02_feature_engineering_y_modelado.ipynb # 5-Fold Stratified CV, métricas KS/PR-AUC y Threshold Moving
│   └── 03_interpretabilidad_shap.ipynb      # Explicabilidad global (Beeswarm) y local (Waterfall)
├── sql/
│   └── 01_analisis_exploratorio.sql         # Consultas analíticas de cohortes y calidad de datos
├── src/
│   ├── preprocessing.py                     # Transformador Scikit-learn sin data leakage
│   └── metrics.py                           # Cálculo de KS, F2-Score y matriz costo-beneficio
├── app.py                                   # Workstation interactiva en Streamlit
└── requirements.txt                         # Dependencias de producción
