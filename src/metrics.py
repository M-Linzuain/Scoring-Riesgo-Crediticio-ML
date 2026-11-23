from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def calcular_ks(y_true: np.ndarray | pd.Series, y_prob: np.ndarray) -> float:
    """
    Calcula el estadístico Kolmogorov-Smirnov (KS), estándar en riesgo crediticio
    para medir la máxima separación entre la distribución acumulada de buenos y malos pagadores.
    """
    prob_positivos = y_prob[np.asarray(y_true) == 1]
    prob_negativos = y_prob[np.asarray(y_true) == 0]
    ks_stat, _ = ks_2samp(prob_positivos, prob_negativos)
    return float(ks_stat)


def evaluar_modelo_crediticio(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    umbral: float = 0.5,
    nombre_modelo: str = "Modelo",
) -> dict[str, float | str]:
    """
    Evalúa un clasificador binario con métricas orientadas a riesgo crediticio
    y desbalance de clases (sin depender de Accuracy).
    """
    y_pred = (y_prob >= umbral).astype(int)

    return {
        "Modelo": nombre_modelo,
        "Umbral": round(umbral, 3),
        "ROC_AUC": round(roc_auc_score(y_true, y_prob), 4),
        "PR_AUC": round(average_precision_score(y_true, y_prob), 4),
        "KS_Stat": round(calcular_ks(y_true, y_prob), 4),
        "Recall_Default": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "Precision_Default": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "F2_Score": round(fbeta_score(y_true, y_pred, beta=2, zero_division=0), 4),
    }


def optimizar_umbral_financiero(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    ganancia_buen_pagador: float = 1000.0,
    perdida_default: float = 5000.0,
) -> tuple[float, pd.DataFrame]:
    """
    Simula el resultado económico neto sobre la cartera para distintos umbrales de corte
    (probabilidad máxima aceptada de default para aprobar el crédito).
    - Verdadero Negativo (TN): Cliente bueno aprobado -> +ganancia_buen_pagador
    - Falso Negativo (FN): Cliente malo aprobado por error -> -perdida_default
    - Positivos predichos (TP + FP): Solicitudes rechazadas -> 0 flujo directo
    """
    umbrales = np.linspace(0.05, 0.80, 76)
    registros = []

    for th in umbrales:
        y_pred = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        tasa_aprobacion = (tn + fn) / len(y_true)
        beneficio_neto = (tn * ganancia_buen_pagador) - (fn * perdida_default)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

        registros.append(
            {
                "umbral": round(float(th), 3),
                "tasa_aprobacion_pct": round(tasa_aprobacion * 100, 2),
                "buenos_aprobados_TN": int(tn),
                "defaults_aprobados_FN": int(fn),
                "defaults_rechazados_TP": int(tp),
                "F2_Score": round(float(f2), 4),
                "beneficio_neto_usd": round(float(beneficio_neto), 2),
            }
        )

    df_simulacion = pd.DataFrame(registros)
    mejor_fila = df_simulacion.loc[df_simulacion["beneficio_neto_usd"].idxmax()]
    return float(mejor_fila["umbral"]), df_simulacion