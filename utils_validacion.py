"""
============================================================================
MÓDULO: VALIDACIÓN AUTOMÁTICA DE DATOS
Valida calidad de datos antes de entrenar modelos SARIMA
============================================================================
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN DE VALIDACIÓN
# ============================================================================

REQUIRED_COLUMNS = ['FECHA-VENTA', 'MODELO3', 'MARCA']
MIN_MONTHS_DATA = 36  # Mínimo 36 meses (3 años) para SARIMA
MAX_MISSING_PCT = 5   # Máximo 5% de datos faltantes
OUTLIER_THRESHOLD = 3  # Desviación estándar para outliers

# ============================================================================
# FUNCIONES DE VALIDACIÓN
# ============================================================================

def validate_dataframe(df, file_name):
    """
    Validación completa de DataFrame
    Retorna: (es_valido, dict_resultados, dict_warnings)
    """
    
    results = {
        'file_name': file_name,
        'valid': True,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'checks': {}
    }
    
    warnings = []
    errors = []
    
    # ========================================
    # CHECK 1: Columnas requeridas
    # ========================================
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    
    if missing_cols:
        errors.append(f"❌ Faltan columnas requeridas: {', '.join(missing_cols)}")
        results['valid'] = False
    else:
        results['checks']['columnas'] = "✅ Todas las columnas requeridas presentes"
    
    # ========================================
    # CHECK 2: Columna de fecha válida
    # ========================================
    if 'FECHA-VENTA' in df.columns:
        try:
            df['FECHA-VENTA'] = pd.to_datetime(df['FECHA-VENTA'], errors='coerce')
            invalid_dates = df['FECHA-VENTA'].isna().sum()
            
            if invalid_dates > 0:
                pct = (invalid_dates / len(df)) * 100
                if pct > MAX_MISSING_PCT:
                    errors.append(f"❌ Demasiadas fechas inválidas: {invalid_dates} ({pct:.1f}%)")
                    results['valid'] = False
                else:
                    warnings.append(f"⚠️ Algunas fechas inválidas: {invalid_dates} ({pct:.1f}%)")
            
            # Verificar rango de fechas
            if df['FECHA-VENTA'].notna().any():
                min_date = df['FECHA-VENTA'].min()
                max_date = df['FECHA-VENTA'].max()
                
                results['checks']['fecha_rango'] = f"✅ Rango: {min_date.strftime('%Y-%m-%d')} a {max_date.strftime('%Y-%m-%d')}"
                
                # Verificar datos incompletos del mes actual
                last_date = df['FECHA-VENTA'].max()
                if last_date.month == datetime.now().month and last_date.year == datetime.now().year:
                    warnings.append(f"⚠️ Datos del mes actual ({last_date.strftime('%B %Y')}) pueden estar incompletos. Recomendación: eliminarlos antes de entrenar.")
        
        except Exception as e:
            errors.append(f"❌ Error procesando fechas: {str(e)}")
            results['valid'] = False
    
    # ========================================
    # CHECK 3: Período temporal suficiente
    # ========================================
    if 'FECHA-VENTA' in df.columns and df['FECHA-VENTA'].notna().any():
        df_temp = df[df['FECHA-VENTA'].notna()].copy()
        df_temp['year_month'] = df_temp['FECHA-VENTA'].dt.to_period('M')
        n_months = df_temp['year_month'].nunique()
        
        results['checks']['meses_datos'] = f"📊 {n_months} meses únicos"
        
        if n_months < MIN_MONTHS_DATA:
            errors.append(f"❌ Datos insuficientes: {n_months} meses (mínimo: {MIN_MONTHS_DATA})")
            results['valid'] = False
        else:
            results['checks']['meses_validacion'] = f"✅ Suficientes datos ({n_months} meses)"
    
    # ========================================
    # CHECK 4: Datos faltantes por columna
    # ========================================
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100
    
    critical_missing = missing_pct[missing_pct > MAX_MISSING_PCT]
    
    if len(critical_missing) > 0:
        warnings.append(f"⚠️ Columnas con >5% faltantes: {', '.join(critical_missing.index.tolist())}")
    
    results['checks']['datos_faltantes'] = f"📊 {missing_data.sum()} valores faltantes totales"
    
    # ========================================
    # CHECK 5: Valores únicos en columnas clave
    # ========================================
    if 'MODELO3' in df.columns:
        unique_models = df['MODELO3'].nunique()
        results['checks']['modelos'] = f"📊 {unique_models} modelos únicos"
    
    if 'MARCA' in df.columns:
        unique_brands = df['MARCA'].nunique()
        results['checks']['marcas'] = f"📊 {unique_brands} marcas únicas"
    
    # ========================================
    # CHECK 6: Outliers en datos numéricos
    # ========================================
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        outliers_count = 0
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                mean = df[col].mean()
                std = df[col].std()
                outliers = ((df[col] - mean).abs() > OUTLIER_THRESHOLD * std).sum()
                outliers_count += outliers
        
        if outliers_count > 0:
            pct = (outliers_count / len(df)) * 100
            if pct > 10:
                warnings.append(f"⚠️ {outliers_count} outliers detectados ({pct:.1f}%)")
            results['checks']['outliers'] = f"📊 {outliers_count} outliers detectados"
    
    # ========================================
    # Consolidar resultados
    # ========================================
    results['errors'] = errors
    results['warnings'] = warnings
    
    return results['valid'], results, warnings, errors

def show_validation_results(results, warnings, errors):
    """Mostrar resultados de validación en UI"""
    
    st.subheader(f"📋 Validación: {results['file_name']}")
    
    # Resumen general
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Filas", f"{results['total_rows']:,}")
    with col2:
        st.metric("Total Columnas", results['total_columns'])
    with col3:
        if results['valid']:
            st.success("✅ VÁLIDO")
        else:
            st.error("❌ INVÁLIDO")
    
    # Checks individuales
    if results['checks']:
        st.markdown("**Verificaciones:**")
        for check, result in results['checks'].items():
            st.markdown(f"- {result}")
    
    # Errores
    if errors:
        st.error("**Errores Críticos:**")
        for error in errors:
            st.markdown(f"- {error}")
    
    # Warnings
    if warnings:
        st.warning("**Advertencias:**")
        for warning in warnings:
            st.markdown(f"- {warning}")
    
    return results['valid']

def preview_data(df, n_rows=10):
    """Preview de los datos con estadísticas"""
    
    st.subheader("👁️ Preview de Datos")
    
    # Primeras filas
    st.markdown(f"**Primeras {n_rows} filas:**")
    st.dataframe(df.head(n_rows), use_container_width=True)
    
    # Estadísticas básicas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Estadísticas Numéricas:**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    with col2:
        st.markdown("**Info de Columnas:**")
        info_df = pd.DataFrame({
            'Columna': df.columns,
            'Tipo': df.dtypes.values,
            'No Nulos': df.count().values,
            'Nulos': df.isnull().sum().values,
            '% Nulos': (df.isnull().sum() / len(df) * 100).round(2).values
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)

def plot_temporal_distribution(df, date_col='FECHA-VENTA'):
    """Gráfico de distribución temporal"""
    
    if date_col not in df.columns:
        return
    
    df_temp = df[df[date_col].notna()].copy()
    
    # Ventas por mes
    df_temp['year_month'] = df_temp[date_col].dt.to_period('M').astype(str)
    ventas_mes = df_temp.groupby('year_month').size().reset_index(name='ventas')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=ventas_mes['year_month'],
        y=ventas_mes['ventas'],
        name='Ventas Mensuales',
        marker_color='#065A82'
    ))
    
    fig.update_layout(
        title='Distribución Temporal de Ventas',
        xaxis_title='Mes',
        yaxis_title='Cantidad de Ventas',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_missing_data(df):
    """Gráfico de datos faltantes"""
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    # Solo columnas con datos faltantes
    missing_df = pd.DataFrame({
        'Columna': missing.index,
        'Faltantes': missing.values,
        'Porcentaje': missing_pct.values
    })
    
    missing_df = missing_df[missing_df['Faltantes'] > 0].sort_values('Faltantes', ascending=False)
    
    if len(missing_df) == 0:
        st.success("✅ No hay datos faltantes")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=missing_df['Columna'],
        y=missing_df['Faltantes'],
        text=missing_df['Porcentaje'].apply(lambda x: f"{x}%"),
        textposition='outside',
        marker_color='red',
        name='Datos Faltantes'
    ))
    
    # Línea de umbral
    fig.add_hline(
        y=len(df) * MAX_MISSING_PCT / 100,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Umbral {MAX_MISSING_PCT}%"
    )
    
    fig.update_layout(
        title='Datos Faltantes por Columna',
        xaxis_title='Columna',
        yaxis_title='Cantidad de Faltantes',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def get_dataset_summary(df):
    """Resumen ejecutivo del dataset"""
    
    summary = {
        'total_registros': len(df),
        'columnas': len(df.columns),
        'memoria_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    if 'FECHA-VENTA' in df.columns:
        df_temp = df[df['FECHA-VENTA'].notna()].copy()
        if len(df_temp) > 0:
            summary['fecha_min'] = df_temp['FECHA-VENTA'].min().strftime('%Y-%m-%d')
            summary['fecha_max'] = df_temp['FECHA-VENTA'].max().strftime('%Y-%m-%d')
            
            df_temp['year_month'] = df_temp['FECHA-VENTA'].dt.to_period('M')
            summary['meses_unicos'] = df_temp['year_month'].nunique()
    
    if 'MODELO3' in df.columns:
        summary['modelos_unicos'] = df['MODELO3'].nunique()
    
    if 'MARCA' in df.columns:
        summary['marcas_unicas'] = df['MARCA'].nunique()
    
    return summary
