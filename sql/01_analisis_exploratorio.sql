-- ============================================================================
-- 01_analisis_exploratorio.sql
-- Auditoría inicial de calidad de datos y análisis de cohortes de riesgo
-- Dataset: Give Me Some Credit (tabla: solicitudes_credito)
-- ============================================================================

-- 1. Resumen general, volumen de registros y tasa base de incumplimiento (Default)
SELECT 
    COUNT(*) AS total_solicitudes,
    SUM(SeriousDlqin2yrs) AS total_defaults,
    ROUND(AVG(SeriousDlqin2yrs) * 100, 2) AS tasa_default_pct,
    COUNT(DISTINCT id_solicitante) AS ids_unicos
FROM solicitudes_credito;

-- 2. Auditoría de valores nulos y anomalías críticas por columna
SELECT
    SUM(CASE WHEN MonthlyIncome IS NULL THEN 1 ELSE 0 END) AS nulos_ingreso_mensual,
    ROUND(AVG(CASE WHEN MonthlyIncome IS NULL THEN 1.0 ELSE 0.0 END) * 100, 2) AS pct_nulos_ingreso,
    SUM(CASE WHEN NumberOfDependents IS NULL THEN 1 ELSE 0 END) AS nulos_dependientes,
    SUM(CASE WHEN age <= 0 OR age < 18 THEN 1 ELSE 0 END) AS edades_invalidas,
    SUM(CASE WHEN RevolvingUtilizationOfUnsecuredLines > 1.0 THEN 1 ELSE 0 END) AS sobreutilizacion_linea,
    SUM(CASE WHEN "NumberOfTime30-59DaysPastDueNotWorse" IN (96, 98) THEN 1 ELSE 0 END) AS codigos_centinela_atrasos
FROM solicitudes_credito;

-- 3. Tasa de Default según severidad máxima de atrasos históricos (CTE + CASE)
WITH perfil_morosidad AS (
    SELECT 
        id_solicitante,
        SeriousDlqin2yrs,
        CASE 
            WHEN NumberOfTimes90DaysLate IN (96, 98) THEN '0_Codigo_Especial_96_98'
            WHEN NumberOfTimes90DaysLate > 0 THEN '4_Atraso_Grave_90+_dias'
            WHEN "NumberOfTime60-89DaysPastDueNotWorse" > 0 THEN '3_Atraso_Moderado_60_89_dias'
            WHEN "NumberOfTime30-59DaysPastDueNotWorse" > 0 THEN '2_Atraso_Leve_30_59_dias'
            ELSE '1_Sin_Atrasos_Previos'
        END AS categoria_historial
    FROM solicitudes_credito
)
SELECT 
    categoria_historial,
    COUNT(*) AS cantidad_clientes,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS participacion_pct,
    SUM(SeriousDlqin2yrs) AS defaults,
    ROUND(AVG(SeriousDlqin2yrs) * 100, 2) AS tasa_default_pct
FROM perfil_morosidad
GROUP BY categoria_historial
ORDER BY categoria_historial;

-- 4. Tasa de Default por tramos etarios
SELECT 
    CASE 
        WHEN age < 18 THEN 'Anomalia (<18)'
        WHEN age BETWEEN 18 AND 29 THEN '18-29 años'
        WHEN age BETWEEN 30 AND 44 THEN '30-44 años'
        WHEN age BETWEEN 45 AND 59 THEN '45-59 años'
        ELSE '60+ años'
    END AS rango_etario,
    COUNT(*) AS total_clientes,
    ROUND(AVG(MonthlyIncome), 2) AS ingreso_promedio,
    ROUND(AVG(SeriousDlqin2yrs) * 100, 2) AS tasa_default_pct
FROM solicitudes_credito
GROUP BY rango_etario
ORDER BY rango_etario;