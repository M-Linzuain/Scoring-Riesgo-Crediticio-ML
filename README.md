# 🏦 Scoring de Riesgo Crediticio y Simulador de Originación (Machine Learning + SHAP)

Sistema *end-to-end* de evaluación de riesgo crediticio construido sobre el dataset **Give Me Some Credit** (150.000 solicitudes reales), integrando auditoría relacional en **SQL (DuckDB)**, ingeniería de características financieras libre de *data leakage*, modelado comparativo con **XGBoost / LightGBM** y despliegue interactivo en **Streamlit** con explicabilidad individual mediante **SHAP**.

---

## 🎯 Problema de Negocio y Enfoque Metodológico

En originación crediticia, el objetivo no es maximizar la exactitud global (*Accuracy*), sino discriminar con precisión qué solicitantes tienen alta probabilidad de incurrir en mora grave (**90 días o más de atraso en los próximos 2 años**, clase minoritaria con prevalencia del **6.68%**) y definir un **punto de corte financiero** que equilibre el margen por buenos pagadores frente al costo de los préstamos incobrables.

### Flujo de Ingeniería
1. **Ingesta Relacional y Auditoría SQL (`sql/01_analisis_exploratorio.sql`):** Carga en DuckDB y análisis de cohortes que detectó el efecto escalonado del historial de mora (desde `2.73%` sin atrasos hasta `41.21%` con mora de 90+ días) y un segmento crítico de códigos centinela (`96`/`98`) con `54.65%` de incumplimiento.
2. **Preprocesamiento Modular Sin *Data Leakage* (`src/preprocessing.py`):** Transformador compatible con `scikit-learn` (`BaseEstimator`, `TransformerMixin`) que aprende medianas y percentiles de acotamiento (*winsorization*) exclusivamente sobre `Train` y construye ratios financieros (`indice_severidad_atrasos`, `gasto_mensual_estimado`, `ingreso_per_capita`).
3. **Validación Cruzada Estratificada y Optimización Económica (`src/metrics.py`):** Evaluación mediante `StratifiedKFold (k=5)` y desplazamiento del umbral de decisión ($0.50 \rightarrow 0.170$) para maximizar el beneficio neto esperado.
4. **Explicabilidad Regulatoria (`notebooks/03_interpretabilidad_shap.ipynb`):** Auditoría global y local con `shap.TreeExplainer`.

---

## 📊 Resultados en Conjunto de Evaluación (Test Set — 30.000 registros)

| Configuración | Umbral | ROC-AUC | PR-AUC | Estadístico KS | Recall (Default) | F2-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Corte Estándar)** | 0.500 | 0.8697 | 0.4084 | 0.5844 | 19.50% | 0.2255 |
| **XGBoost (Corte Financiero Óptimo)** | **0.170** | **0.8697** | **0.4084** | **0.5844** | **56.36%** | **0.5097** |

---

## 🚀 Ejecución Local

```bash
# 1. Clonar el repositorio
git clone [https://github.com/M-Linzuain/Scoring-Riesgo-Crediticio-ML.git](https://github.com/M-Linzuain/Scoring-Riesgo-Crediticio-ML.git)
cd Scoring-Riesgo-Crediticio-ML

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la aplicación interactiva
streamlit run app.py