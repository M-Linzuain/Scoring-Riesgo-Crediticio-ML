from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CreditRiskPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transformador compatible con Scikit-learn para limpieza estructural,
    imputación sin data leakage e ingeniería de características financieras
    sobre el dataset Give Me Some Credit.
    """

    def __init__(self, cap_utilization: float = 2.0, cap_debt_ratio_quantile: float = 0.99):
        self.cap_utilization = cap_utilization
        self.cap_debt_ratio_quantile = cap_debt_ratio_quantile

        # Atributos aprendidos exclusivamente en fit()
        self.median_age_: float = 0.0
        self.median_income_: float = 0.0
        self.median_dependents_: float = 0.0
        self.debt_ratio_upper_bound_: float = 0.0
        self.feature_names_out_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> CreditRiskPreprocessor:
        df = X.copy()

        # 1. Mediana de edad válida (>= 18 años)
        valid_ages = df.loc[df["age"].astype(float) >= 18, "age"].astype(float)
        self.median_age_ = float(valid_ages.median())

        # 2. Medianas de variables con nulos estructurales
        self.median_income_ = float(df["MonthlyIncome"].astype(float).median())
        self.median_dependents_ = float(df["NumberOfDependents"].astype(float).median())

        # 3. Umbral superior para DebtRatio calculado solo sobre registros con ingreso declarado > 0
        income_num = df["MonthlyIncome"].astype(float).fillna(0)
        valid_debt_ratios = df.loc[income_num > 0, "DebtRatio"].astype(float)
        self.debt_ratio_upper_bound_ = float(valid_debt_ratios.quantile(self.cap_debt_ratio_quantile))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Eliminar identificador si está presente
        if "id_solicitante" in df.columns:
            df = df.drop(columns=["id_solicitante"])

        # Estandarizamos todas las columnas a float64 para evitar conflictos con dtypes 'Int64' de DuckDB
        df = df.astype(float)

        cols_atrasos = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
        ]

        # --- 1. Banderas de información faltante y códigos centinela ---
        df["flag_ingreso_no_declarado"] = df["MonthlyIncome"].isna().astype(int)
        df["flag_dependientes_nulos"] = df["NumberOfDependents"].isna().astype(int)
        df["flag_codigo_especial_atraso"] = (
            df[cols_atrasos].isin([96, 98]).any(axis=1).astype(int)
        )

        # --- 2. Limpieza de anomalías estructurales ---
        # Edades menores a 18 (ej. age == 0) se imputan con la mediana aprendida en train
        df.loc[df["age"] < 18, "age"] = self.median_age_

        # Reemplazar códigos especiales 96/98 en atrasos por 0 tras capturar la bandera de riesgo extremo
        for col in cols_atrasos:
            df[col] = df[col].replace({96: np.nan, 98: np.nan}).fillna(0.0)

        # --- 3. Imputación de nulos ---
        df["MonthlyIncome"] = df["MonthlyIncome"].fillna(self.median_income_)
        df["NumberOfDependents"] = df["NumberOfDependents"].fillna(self.median_dependents_)

        # --- 4. Acotamiento (Winsorization) de ratios extremos ---
        df["RevolvingUtilization_clean"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(
            lower=0.0, upper=self.cap_utilization
        )
        df["flag_sobreutilizacion_100"] = (
            df["RevolvingUtilizationOfUnsecuredLines"] > 1.0
        ).astype(int)

        df["DebtRatio_clean"] = df["DebtRatio"].clip(
            lower=0.0, upper=self.debt_ratio_upper_bound_
        )

        # --- 5. Ingeniería de características financieras ---
        df["gasto_mensual_estimado"] = df["DebtRatio_clean"] * df["MonthlyIncome"]
        df["ingreso_per_capita"] = df["MonthlyIncome"] / (1.0 + df["NumberOfDependents"])

        df["total_atrasos_historicos"] = (
            df["NumberOfTime30-59DaysPastDueNotWorse"]
            + df["NumberOfTime60-89DaysPastDueNotWorse"]
            + df["NumberOfTimes90DaysLate"]
        )

        df["indice_severidad_atrasos"] = (
            1.0 * df["NumberOfTime30-59DaysPastDueNotWorse"]
            + 2.0 * df["NumberOfTime60-89DaysPastDueNotWorse"]
            + 3.0 * df["NumberOfTimes90DaysLate"]
        )

        df["ratio_lineas_inmobiliarias"] = df["NumberRealEstateLoansOrLines"] / (
            1.0 + df["NumberOfOpenCreditLinesAndLoans"]
        )

        # Eliminamos las columnas crudas sin acotar
        df = df.drop(columns=["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"])

        self.feature_names_out_ = df.columns.tolist()
        return df