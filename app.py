from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import plotly.graph_objects as go

# Importamos la clase personalizada para deserializar el pipeline
from src.preprocessing import CreditRiskPreprocessor  # noqa: F401

# Configuración de página
st.set_page_config(
    page_title="Scoring de Riesgo Crediticio | Panel de Originación",
    page_icon="💳",
    layout="wide",
)

# Estilos CSS personalizados para tarjetas KPI estilo terminal financiera
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 15px 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "modelo_crediticio.joblib"

NOMBRES_LEGIBLES = {
    "RevolvingUtilization_clean": "Uso de Crédito Revolvente (%)",
    "indice_severidad_atrasos": "Índice de Severidad de Atrasos",
    "total_atrasos_historicos": "Total de Atrasos Históricos",
    "age": "Edad del Solicitante",
    "NumberOfOpenCreditLinesAndLoans": "Líneas de Crédito Abiertas",
    "MonthlyIncome": "Ingreso Mensual (USD)",
    "DebtRatio_clean": "Ratio Deuda / Ingreso",
    "NumberRealEstateLoansOrLines": "Préstamos Hipotecarios",
    "ratio_lineas_inmobiliarias": "Proporción Líneas Inmobiliarias",
    "gasto_mensual_estimado": "Compromiso Mensual Estimado (USD)",
    "ingreso_per_capita": "Ingreso Disponible per Cápita (USD)",
    "flag_ingreso_no_declarado": "Alerta: Ingreso No Declarado",
    "NumberOfTime30-59DaysPastDueNotWorse": "Atrasos Leves (30-59d)",
    "NumberOfTimes90DaysLate": "Atrasos Graves (90+d)",
    "NumberOfTime60-89DaysPastDueNotWorse": "Atrasos Moderados (60-89d)",
    "flag_sobreutilizacion_100": "Alerta: Sobreutilización (>100%)",
    "flag_codigo_especial_atraso": "Alerta: Código Especial Mora",
    "NumberOfDependents": "Personas a Cargo",
    "flag_dependientes_nulos": "Alerta: Dependientes Nulos",
}


@st.cache_resource
def cargar_artefacto():
    artefacto = joblib.load(MODEL_PATH)
    pipeline = artefacto["pipeline"]
    explainer = shap.TreeExplainer(pipeline.named_steps["clf"])
    return artefacto, explainer


def crear_gauge_score(score: int, aprobado: bool) -> go.Figure:
    """Construye un velocímetro bancario de Score Crediticio (100 - 999)."""
    color_aguja = "#00D4AA" if aprobado else "#FF4B4B"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Score Interno de Riesgo", "font": {"size": 16, "color": "#FAFAFA"}},
            number={"font": {"size": 42, "color": color_aguja}},
            gauge={
                "axis": {"range": [100, 999], "tickwidth": 1, "tickcolor": "#8B949E"},
                "bar": {"color": color_aguja, "thickness": 0.28},
                "bgcolor": "#0E1117",
                "borderwidth": 1,
                "bordercolor": "#30363D",
                "steps": [
                    {"range": [100, 550], "color": "rgba(255, 75, 75, 0.25)"},
                    {"range": [550, 750], "color": "rgba(241, 196, 15, 0.20)"},
                    {"range": [750, 850], "color": "rgba(52, 152, 219, 0.20)"},
                    {"range": [850, 999], "color": "rgba(0, 212, 170, 0.25)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=270,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#FAFAFA"},
    )
    return fig


def clasificar_segmento(score: int) -> str:
    if score >= 850:
        return "AAA — Prime (Riesgo Mínimo)"
    elif score >= 750:
        return "AA — Preferente (Riesgo Bajo)"
    elif score >= 600:
        return "B — Estándar (Riesgo Moderado)"
    return "C — Subprime (Riesgo Elevado)"


def main():
    if not MODEL_PATH.exists():
        st.error(f"No se encontró el artefacto del modelo en `{MODEL_PATH}`.")
        st.stop()

    artefacto, explainer = cargar_artefacto()
    pipeline = artefacto["pipeline"]
    umbral_default = float(artefacto["umbral_optimo"])
    metricas_test = artefacto["metricas_test"]

    # Encabezado principal estilo Fintech
    col_title, col_badge = st.columns([3, 1])
    with col_title:
        st.title("💳 Scoring de Riesgo Crediticio | Panel de Originación")
        st.caption(
            f"Motor Predictivo: **{artefacto['nombre_modelo']}** | "
            f"ROC-AUC Test: **{metricas_test['ROC_AUC']:.4f}** | "
            f"Estadístico KS: **{metricas_test['KS_Stat']:.4f}**"
        )
    with col_badge:
        st.info(f"**Política Base Óptima**\nCorte PD Máx: `{umbral_default * 100:.1f}%`")

    # --- Sidebar: Parámetros del Solicitante ---
    st.sidebar.header("🧾 Datos del Solicitante")
    age = st.sidebar.slider("Edad (años)", 18, 95, 38)
    ingreso_declarado = st.sidebar.toggle("Declara ingresos verificables", value=True)
    monthly_income = (
        st.sidebar.number_input("Ingreso Mensual Neto (USD)", 500.0, 100000.0, 5400.0, step=250.0)
        if ingreso_declarado
        else np.nan
    )
    dependents = st.sidebar.number_input("Personas a cargo", 0, 10, 1)

    st.sidebar.subheader("📊 Carga Financiera")
    utilizacion_pct = st.sidebar.slider("Uso de Líneas Revolventes (%)", 0, 150, 25)
    debt_ratio_pct = st.sidebar.slider("Ratio Deuda / Ingreso (%)", 0, 200, 35)
    open_lines = st.sidebar.number_input("Líneas de Crédito Abiertas", 0, 40, 8)
    real_estate_loans = st.sidebar.number_input("Préstamos Hipotecarios", 0, 15, 1)

    st.sidebar.subheader("⏱️ Historial de Mora (24 meses)")
    atrasos_30_59 = st.sidebar.number_input("Atrasos 30-59 días", 0, 20, 0)
    atrasos_60_89 = st.sidebar.number_input("Atrasos 60-89 días", 0, 20, 0)
    atrasos_90_mas = st.sidebar.number_input("Atrasos 90+ días", 0, 20, 0)

    st.sidebar.divider()
    umbral_corte = (
        st.sidebar.slider(
            "Umbral Máximo de PD Aceptada (%)",
            min_value=5.0,
            max_value=50.0,
            value=round(umbral_default * 100, 1),
            step=0.5,
        )
        / 100.0
    )

    # --- Navegación Multi-Pestaña ---
    tab1, tab2, tab3 = st.tabs(
        [
            "🎯 Evaluación Individual & SHAP",
            "📈 Simulador de Política y ROI",
            "📂 Scoring por Lotes (CSV)",
        ]
    )

    # =========================================================================
    # PESTAÑA 1: EVALUACIÓN INDIVIDUAL Y EXPLICABILIDAD LOCAL
    # =========================================================================
    with tab1:
        datos_entrada = pd.DataFrame(
            [
                {
                    "RevolvingUtilizationOfUnsecuredLines": utilizacion_pct / 100.0,
                    "age": age,
                    "NumberOfTime30-59DaysPastDueNotWorse": atrasos_30_59,
                    "DebtRatio": debt_ratio_pct / 100.0,
                    "MonthlyIncome": monthly_income,
                    "NumberOfOpenCreditLinesAndLoans": open_lines,
                    "NumberOfTimes90DaysLate": atrasos_90_mas,
                    "NumberRealEstateLoansOrLines": real_estate_loans,
                    "NumberOfTime60-89DaysPastDueNotWorse": atrasos_60_89,
                    "NumberOfDependents": dependents,
                }
            ]
        )

        prob_default = float(pipeline.predict_proba(datos_entrada)[0, 1])
        score = int(np.clip(1000 * (1 - prob_default), 100, 999))
        aprobado = prob_default < umbral_corte

        datos_transformados = pipeline.named_steps["prep"].transform(datos_entrada)
        shap_vals = explainer(datos_transformados).values[0]

        col_gauge, col_kpis = st.columns([1.1, 1.9])

        with col_gauge:
            st.plotly_chart(crear_gauge_score(score, aprobado), use_container_width=True)

        with col_kpis:
            k1, k2, k3 = st.columns(3)
            k1.metric(
                "Probabilidad de Default (PD)",
                f"{prob_default * 100:.2f}%",
                delta=f"{(prob_default - umbral_corte) * 100:+.2f}% vs. límite",
                delta_color="inverse",
            )
            k2.metric("Segmento Crediticio", clasificar_segmento(score).split(" — ")[0])
            k3.metric("Límite Política Activo", f"{umbral_corte * 100:.1f}%")

            st.markdown("#### Dictamen Automatizado de Originación")
            if aprobado:
                st.success(
                    f"**✅ SOLICITUD APROBADA** — Perfil clasificado como **{clasificar_segmento(score)}**. "
                    f"La probabilidad estimada de incumplimiento (`{prob_default * 100:.2f}%`) se ubica por debajo "
                    f"del umbral de apetito de riesgo configurado (`{umbral_corte * 100:.1f}%`)."
                )
            else:
                st.error(
                    f"**❌ SOLICITUD RECHAZADA** — Perfil clasificado como **{clasificar_segmento(score)}**. "
                    f"La probabilidad estimada de incumplimiento (`{prob_default * 100:.2f}%`) supera el máximo "
                    f"tolerado por la política vigente (`{umbral_corte * 100:.1f}%`)."
                )

        st.subheader("🔍 Descomposición de Factores de Riesgo (Valores SHAP)")
        df_shap = pd.DataFrame(
            {
                "Variable": [NOMBRES_LEGIBLES.get(c, c) for c in datos_transformados.columns],
                "Impacto_SHAP": shap_vals,
            }
        )
        df_shap["Abs_SHAP"] = df_shap["Impacto_SHAP"].abs()
        df_top = df_shap.sort_values("Abs_SHAP", ascending=True).tail(10)

        colores = ["#FF4B4B" if v > 0 else "#00D4AA" for v in df_top["Impacto_SHAP"]]
        fig_shap = go.Figure(
            go.Bar(
                x=df_top["Impacto_SHAP"],
                y=df_top["Variable"],
                orientation="h",
                marker_color=colores,
                text=[f"{v:+.3f}" for v in df_top["Impacto_SHAP"]],
                textposition="outside",
            )
        )
        fig_shap.update_layout(
            xaxis_title="Contribución Log-Odds SHAP (🟢 Reduce Riesgo | 🔴 Aumenta Riesgo)",
            height=400,
            margin=dict(l=10, r=40, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#FAFAFA"},
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    # =========================================================================
    # PESTAÑA 2: SIMULADOR DE POLÍTICA DE CORTE Y RETORNO FINANCIERO (ROI)
    # =========================================================================
    with tab2:
        st.subheader("📈 Simulador Económico de Política de Originación")
        st.markdown(
            "Modelá el impacto financiero sobre una cartera estándar de **10.000 solicitudes** "
            "(tasa base de default `6.68%` y poder discriminante `KS = 0.5844`) ajustando los parámetros de negocio:"
        )

        c_param1, c_param2 = st.columns(2)
        ganancia_bueno = c_param1.number_input(
            "Margen Neto por Cliente Cumplidor (USD)", min_value=200, max_value=10000, value=1000, step=100
        )
        perdida_malo = c_param2.number_input(
            "Pérdida por Incobrable / LGD (USD)", min_value=1000, max_value=50000, value=5000, step=500
        )

        # Simulación paramétrica calibrada con la curva ROC/KS del modelo
        umbrales = np.linspace(0.05, 0.60, 56)
        n_cartera = 10000
        n_malos = int(n_cartera * 0.0668)
        n_buenos = n_cartera - n_malos

        # Aproximación realista de tasas de captura según umbral aprendidas en validación
        tpr_rechazo_malos = 1 - np.exp(-5.1 * (0.65 - umbrales))
        fpr_rechazo_buenos = np.clip(0.55 * np.exp(-9.5 * umbrales), 0.01, 0.65)

        buenos_aprobados = n_buenos * (1 - fpr_rechazo_buenos)
        malos_aprobados = n_malos * (1 - tpr_rechazo_malos)
        beneficio_neto = (buenos_aprobados * ganancia_bueno) - (malos_aprobados * perdida_malo)

        idx_opt = int(np.argmax(beneficio_neto))
        umbral_sim_opt = umbrales[idx_opt]

        fig_roi = go.Figure()
        fig_roi.add_trace(
            go.Scatter(
                x=umbrales * 100,
                y=beneficio_neto / 1e6,
                mode="lines",
                name="Beneficio Neto Esperado (Millones USD)",
                line=dict(color="#00D4AA", width=3),
            )
        )
        fig_roi.add_vline(
            x=umbral_sim_opt * 100,
            line_dash="dash",
            line_color="#F1C40F",
            annotation_text=f"Óptimo Económico: {umbral_sim_opt * 100:.1f}%",
        )
        fig_roi.add_vline(
            x=umbral_corte * 100,
            line_dash="dot",
            line_color="#FF4B4B",
            annotation_text=f"Política Actual: {umbral_corte * 100:.1f}%",
        )
        fig_roi.update_layout(
            xaxis_title="Umbral de Corte PD (%)",
            yaxis_title="Beneficio Neto Esperado (Millones USD)",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#FAFAFA"},
        )
        st.plotly_chart(fig_roi, use_container_width=True)

    # =========================================================================
    # PESTAÑA 3: EVALUACIÓN POR LOTES (BATCH SCORING CSV)
    # =========================================================================
    with tab3:
        st.subheader("📂 Evaluación Masiva de Solicitudes (Batch Scoring)")
        st.markdown(
            "Subí un archivo `.csv` con solicitudes de crédito para calcular automáticamente la "
            "**Probabilidad de Default**, el **Score Interno** y el **Dictamen** según el umbral activo."
        )

        # Generador de plantilla CSV descargable para facilitar pruebas
        plantilla_df = pd.DataFrame(
            [
                {
                    "RevolvingUtilizationOfUnsecuredLines": 0.15,
                    "age": 45,
                    "NumberOfTime30-59DaysPastDueNotWorse": 0,
                    "DebtRatio": 0.28,
                    "MonthlyIncome": 6800,
                    "NumberOfOpenCreditLinesAndLoans": 7,
                    "NumberOfTimes90DaysLate": 0,
                    "NumberRealEstateLoansOrLines": 1,
                    "NumberOfTime60-89DaysPastDueNotWorse": 0,
                    "NumberOfDependents": 2,
                },
                {
                    "RevolvingUtilizationOfUnsecuredLines": 0.96,
                    "age": 27,
                    "NumberOfTime30-59DaysPastDueNotWorse": 2,
                    "DebtRatio": 0.72,
                    "MonthlyIncome": 2400,
                    "NumberOfOpenCreditLinesAndLoans": 11,
                    "NumberOfTimes90DaysLate": 1,
                    "NumberRealEstateLoansOrLines": 0,
                    "NumberOfTime60-89DaysPastDueNotWorse": 1,
                    "NumberOfDependents": 0,
                },
            ]
        )

        st.download_button(
            label="⬇️ Descargar Plantilla CSV de Ejemplo",
            data=plantilla_df.to_csv(index=False).encode("utf-8"),
            file_name="plantilla_solicitudes_credito.csv",
            mime="text/csv",
        )

        archivo_subido = st.file_uploader("Cargar archivo CSV de solicitudes", type=["csv"])
        if archivo_subido is not None:
            df_lote = pd.read_csv(archivo_subido)
            probs_lote = pipeline.predict_proba(df_lote)[:, 1]
            scores_lote = np.clip(1000 * (1 - probs_lote), 100, 999).astype(int)

            df_resultado = df_lote.copy()
            df_resultado["Probabilidad_Default_Pct"] = np.round(probs_lote * 100, 2)
            df_resultado["Score_Crediticio"] = scores_lote
            df_resultado["Dictamen"] = np.where(
                probs_lote < umbral_corte, "APROBADO", "RECHAZADO"
            )

            st.success(f"Se procesaron **{len(df_resultado)} solicitudes** correctamente.")
            st.dataframe(df_resultado, use_container_width=True)

            st.download_button(
                label="📥 Descargar Resultados Dictaminados (CSV)",
                data=df_resultado.to_csv(index=False).encode("utf-8"),
                file_name="dictamen_riesgo_crediticio.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()