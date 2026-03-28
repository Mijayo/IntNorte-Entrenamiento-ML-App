# Sistema TIGGO 2 — Predicción de Ventas con ML

Sistema multipage en Streamlit para entrenar modelos SARIMA, comparar múltiples algoritmos de predicción (SARIMA, Prophet, Regresión Lineal, Random Forest, XGBoost) y visualizar predicciones de demanda del Chery Tiggo 2. Desplegado en **Streamlit Cloud** con almacenamiento persistente en **Supabase Storage** y control de acceso por roles.

---

## Arquitectura

```
app_principal.py              ← Entry point (autenticación + página de inicio)
pages/
├── 1_Entrenamiento.py        ← Entrenamiento SARIMA (Admin / Analista)
├── 2_Dashboard.py            ← Dashboard de negocio (todos los roles)
└── 3_Comparativa_ML.py       ← Comparativa de 5 modelos ML (Admin / Analista)
auth_system.py                ← Autenticación, sesiones y show_header()
supabase_io.py                ← Capa de I/O centralizada (Supabase Storage)
utils_validacion.py           ← Validación de datos de entrada
requirements.txt
.streamlit/
├── secrets.toml              ← Credenciales reales (NO en el repo)
└── secrets.toml.example      ← Plantilla (sí en el repo)
```

> No hay carpetas locales de datos. Todos los artefactos del modelo se guardan y leen desde **Supabase Storage** (`bucket: modelos-ml`).

---

## Estructura del bucket Supabase (`modelos-ml`)

```
latest.txt                          ← Apunta al run de producción activo
training_log.json                   ← Historial completo de entrenamientos
YYYYMMDD_HHMMSS/                    ← Una carpeta por run de entrenamiento
    metricas_mejoradas.json
    prediccion_total_mejorada.xlsx
    grid_search_results.xlsx
    walk_forward_validation.xlsx
    historico_total_mejorado.xlsx
    modelo_total_mejorado.pkl.gz
    acf_plot.png
    pacf_plot.png
```

---

## Requisitos

```
streamlit, pandas, numpy, statsmodels, scikit-learn,
matplotlib, plotly, pillow, openpyxl, supabase, google-genai,
prophet, optuna, xgboost
```

```bash
pip install -r requirements.txt
```

---

## Configuración

### 1. Credenciales

Copia la plantilla y rellena los valores reales:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Editar con URL, anon key de Supabase y contraseñas de usuarios
```

> **Nunca subas `.streamlit/secrets.toml` al repositorio.**

Estructura esperada en `secrets.toml`:

```toml
# Clave API de Google Gemini (para el Asistente IA del Dashboard)
GENAI_API_KEY = "..."

[supabase]
url    = "https://<proyecto>.supabase.co"
key    = "SUPABASE_ANON_KEY"
bucket = "modelos-ml"

[users.admin]
password = "..."
role = "admin"
name = "Administrador"
icon = "👑"

[users.admin.permissions]
entrenar_modelos = true
ver_predicciones = true
exportar = true
# ... ver secrets.toml.example para la lista completa
```

### 2. Ejecutar en local

```bash
streamlit run app_principal.py
```

### 3. Desplegar en Streamlit Cloud

1. Conectar el repositorio en [share.streamlit.io](https://share.streamlit.io)
2. Entry point: `app_principal.py`
3. Añadir el contenido de `secrets.toml` en **App settings → Secrets**

---

## Roles y permisos

| Rol | Entrenamiento | Comparativa ML | Tabs del Dashboard |
|-----|:-------------:|:--------------:|---------------------|
| `admin` | ✅ | ✅ | Dashboard, Predicciones, ACF/PACF, Grid Search, Walk-Forward, Métricas técnicas, **Asistente IA**, Concesionarios |
| `analyst` | ✅ | ✅ | Dashboard, Predicciones, ACF/PACF, Grid Search, Walk-Forward, Métricas técnicas, **Asistente IA**, Concesionarios |
| `manager` | — | — | Dashboard, Predicciones, Recomendaciones de compra, **Asistente IA**, Concesionarios |
| `viewer` | — | — | Dashboard, Predicciones |

---

## Flujo de trabajo

```
1. [Entrenamiento]  Cargar el Excel con el histórico de ventas
2. [Entrenamiento]  Limpieza automática: duplicados por CHASIS + filas sin MODELO3
3. [Entrenamiento]  Validación automática de calidad de datos
4. [Entrenamiento]  Test ADF de estacionariedad
5. [Entrenamiento]  Búsqueda Optuna (TPE) → mejores parámetros SARIMA (criterio: MAPE mínimo)
6. [Entrenamiento]  Walk-forward validation sobre los últimos N meses
7. [Entrenamiento]  Modelo final + forecast generado
8. [Entrenamiento]  Artefactos subidos automáticamente a Supabase Storage
9. [Entrenamiento]  Comparación con el modelo de producción actual
10. [Entrenamiento] Clic en "Aprobar" → latest.txt actualizado → Dashboard activo
11. [Dashboard]     Carga el modelo activo automáticamente al arrancar
12. [Dashboard]     Barra lateral permite cambiar entre cualquier run histórico
13. [Comparativa]   Carga el mismo histórico y enfrenta 5 modelos en un solo clic
```

Sin ZIPs. Sin copias manuales. El entrenamiento escribe directamente en Supabase y el dashboard lee desde allí.

---

## App 1 — Entrenamiento (`pages/1_Entrenamiento.py`)

### Formato de datos de entrada

El uploader detecta automáticamente el tipo de archivo según las hojas disponibles:

| Tipo | Hoja requerida | Se guarda en |
|------|---------------|--------------|
| Ventas | `Hoja1` | `session_state['df_raw']` |
| Stock | `Stock Actual` | `session_state['df_stock']` |

Se pueden subir varios archivos de ambos tipos en una misma carga.

**Columnas esperadas en el Excel de ventas (`Hoja1`):**

| Columna | Descripción |
|---------|-------------|
| `FECHA-VENTA` | Fecha de venta (parseable por pandas) |
| `MARCA` | Marca del vehículo |
| `MODELO3` | Nombre del modelo |

### Pestañas de la app de Entrenamiento

| # | Pestaña | Descripción |
|---|---------|-------------|
| 1 | 📤 Cargar Datos | Subida de Excel de ventas y/o stock |
| 2 | ✅ Validación | Calidad de datos, preview, distribución temporal |
| 3 | 🎓 Preparar Datos | Pipeline académico paso a paso + descarga del `.xlsx` de entrenamiento |
| 4 | 🤖 Entrenamiento | Configuración, búsqueda Optuna (TPE), walk-forward, guardado en Supabase |
| 5 | 📊 Comparación | Métricas nuevo vs. actual, residuos, botón de aprobación |
| 6 | 📋 Historial | Log de todos los entrenamientos con evolución del MAPE |

### Parámetros configurables (pestaña Entrenamiento)

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| Marca | `CHERY` | Filtro de marca |
| Modelo | `TIGGO 2` | Filtro de modelo |
| Fecha inicio | `2021-01-01` | Ignorar ventas anteriores a esta fecha |
| Fecha fin de datos | hoy | Límite superior del histórico usado para entrenar. Se combina con «Eliminar mes actual»: gana el corte más conservador |
| Horizonte | `6` meses | Meses a predecir (3–12) |
| Máx. ventas | `100` unid./mes | Límite superior de predicciones válidas |
| Excluir mes actual | `true` | Eliminar el mes en curso (datos incompletos) |

### Pestaña 🎓 Preparar Datos

Modo académico que muestra el pipeline completo de transformación de datos, sin necesidad de lanzar un entrenamiento:

1. **Datos brutos** — muestra de las filas individuales del Excel
2. **Filtro marca** — descarte de filas de otras marcas
3. **Filtro modelo** — aislamiento del modelo objetivo
4. **Rango de fechas** — recorte temporal configurable
5. **Resample mensual** — conversión de filas individuales a serie temporal (`resample('ME').size()`) con tabla y gráfico
6. **Variable exógena** — ventas mensuales de los demás modelos de la marca, usadas como regresora en SARIMAX

Al final de la pestaña hay un **botón de descarga** del `.xlsx` resultante con tres hojas: `Serie_SARIMA`, `Ventas_Mensuales` y `Comparativa`.

### Modelo SARIMA

- **Algoritmo**: SARIMAX con variable exógena (ventas de otros modelos de la misma marca)
- **Búsqueda de hiperparámetros**: [Optuna](https://optuna.org/) con sampler **TPE (Tree-structured Parzen Estimator)** — 80 trials bayesianos sobre el espacio `p ∈ {0–3}`, `d ∈ {0–1}`, `q ∈ {0–3}`, `P ∈ {0–1}`, `D ∈ {0–1}`, `Q ∈ {0–2}`, `m=12` (~4× más rápido que el grid search exhaustivo de 384 combinaciones, con igual o mejor calidad)
- **Criterio**: MAPE mínimo sobre el conjunto de test; se descartan predicciones fuera del rango `[0, max_ventas]`
- **Trials descartados**: combinaciones con predicciones negativas, superiores al límite configurado o con errores numéricos de convergencia
- **Variable exógena**: ventas mensuales de los demás modelos de la misma marca (`ventas_otros`)
- **Intervalos de confianza**: 95% en todos los puntos del forecast

### Flujo de aprobación

La pestaña **Comparación** muestra métricas lado a lado con el modelo de producción (MAPE, AIC, próximo mes). Al hacer clic en **Aprobar** se actualiza `latest.txt` en Supabase y el Dashboard refleja el cambio inmediatamente. Los runs no aprobados quedan en el historial sin afectar producción.

---

## App 3 — Comparativa ML (`pages/3_Comparativa_ML.py`)

Página de comparación que enfrenta hasta **5 modelos** sobre el mismo histórico mensual de ventas del Tiggo 2. Accesible solo para `admin` y `analyst`.

### Modelos disponibles

| Modelo | Tipo | Enfoque |
|--------|------|---------|
| **SARIMA** | Serie de tiempo | Parámetros (p,d,q)(P,D,Q)₁₂ configurables manualmente |
| **Prophet** | Serie de tiempo | Estacionalidad multiplicativa anual + festivos de México (MX) |
| **Regresión Lineal** | ML supervisado | Lag features + rolling stats + calendario |
| **Random Forest** | ML supervisado | 300 estimadores, captura relaciones no lineales |
| **XGBoost** | ML supervisado | Gradient boosting, lr=0.05, max_depth=4 |

### Feature engineering para modelos ML

Los modelos ML (Regresión Lineal, Random Forest, XGBoost) se alimentan de features derivadas de la propia serie temporal:

| Feature | Descripción |
|---------|-------------|
| `lag_1` … `lag_12` | Ventas de 1, 2, 3, 6 y 12 meses atrás |
| `roll_mean_3` / `roll_mean_6` | Media móvil de los últimos 3 y 6 meses (desplazada 1 período) |
| `roll_std_3` | Desviación estándar móvil de 3 meses |
| `mes` | Mes del año (1–12) |
| `trimestre` | Trimestre del año (1–4) |

> Los modelos ML requieren al menos **`12 + n_test + 5`** meses de histórico para tener un conjunto de entrenamiento estable (el lag de 12 meses consume las primeras 12 observaciones).

### Flujo de la comparación

| Paso | Descripción |
|------|-------------|
| 1. Fuente de datos | Carga el histórico desde un run guardado en Supabase **o** sube un Excel propio |
| 2. Configuración | Meses de test (hold-out 3–12), selección de modelos a incluir, parámetros SARIMA y festivos Prophet |
| 3. Ejecutar | Entrena cada modelo seleccionado con barra de progreso y tiempo de ejecución |
| 4. Resultados | Tabla de métricas, ganador por MAPE, gráficas y tabla detallada descargable |

### Métricas comparadas

| Métrica | Descripción | Criterio |
|---------|-------------|---------|
| **MAE** | Error absoluto medio (unidades) | Menor |
| **RMSE** | Raíz del error cuadrático medio | Menor |
| **MAPE (%)** | Error porcentual medio — criterio principal | Menor |
| **R²** | Proporción de varianza explicada | Mayor (máx. 1.0) |
| **Tiempo (s)** | Segundos de entrenamiento | Menor |

La tabla de resultados resalta en verde las mejores celdas de cada métrica. El ganador se anuncia por MAPE mínimo.

### Gráficas de resultados

1. **Predicciones vs Real** — histórico train + real test + todos los modelos en el mismo eje, cada uno con su color
2. **Error absoluto por mes** — barras agrupadas para comparar el error mes a mes entre modelos
3. **Importancia de features** — solo para modelos ML (Gini para RF/XGBoost, |coeficiente| para Regresión Lineal)

### Descarga de resultados

Botón de descarga del período de test como **CSV**, con columnas `Real`, predicción y error absoluto de cada modelo.

---

## App 2 — Dashboard (`pages/2_Dashboard.py`)

### Selector de versión

La barra lateral lista todos los runs disponibles ordenados por fecha. El run activo (apuntado por `latest.txt`) aparece con 🟢. Los históricos con 🔵 pueden seleccionarse sin alterar producción.

### Carga de datos de concesionarios (sidebar)

El expander **📂 Datos de Concesionarios** en la barra lateral acepta un Excel con el histórico de ventas del mercado. El archivo se normaliza automáticamente:

- Columnas de fecha aceptadas: `FECHA_VENTA`, `FECHA-VENTA`, `FECHA VENTA`
- Columnas de modelo aceptadas: `MODELO2`, `MODELO3`, `MODELO`
- Columna de concesionario: `DET_CC` (prioridad) o `AGE`
- Columna de ciudad: `AGE` o `CIUDAD`/`REGION`
- Si la primera fila contiene descripciones (todo texto), se descarta automáticamente

### Tab 🏪 Concesionarios (admin / analista / gerente)

Disponible una vez cargado el Excel desde el sidebar. Muestra análisis de ventas CHERY filtradas por año, modelo y ciudad:

| Elemento | Descripción |
|----------|-------------|
| KPIs | Total ventas CHERY, nº de concesionarios, top concesionario, modelo más vendido |
| Barras horizontales | Ventas totales por concesionario, coloreadas por ciudad |
| Evolución mensual | Serie temporal por concesionario (multiselect, hasta 5 por defecto) |
| Distribución modelos | Barras apiladas modelos × concesionario |
| Ranking | Tabla con ventas, % total y % acumulado (análisis ABC) |

### Tabs por rol

| Tab | Admin | Analista | Gerente | Viewer |
|-----|:-----:|:--------:|:-------:|:------:|
| Dashboard (KPIs + histórico) | ✅ | ✅ | ✅ | ✅ |
| Predicciones (N meses + IC) | ✅ | ✅ | ✅ | ✅ |
| Recomendaciones de compra | — | — | ✅ | — |
| Análisis ACF/PACF | ✅ | ✅ | — | — |
| Resultados Grid Search | ✅ | ✅ | — | — |
| Walk-forward validation | ✅ | ✅ | — | — |
| Métricas técnicas completas | ✅ | ✅ | — | — |
| **Asistente IA (Gemini)** | ✅ | ✅ | ✅ | — |
| Ventas por Concesionario | ✅ | ✅ | ✅ | — |

### Tab 🤖 Asistente IA (admin / analista / gerente)

Chat sobre el modelo entrenado, alimentado con Gemini (`gemini-2.5-flash`). El contexto que recibe el LLM incluye parámetros SARIMA, AIC/BIC, MAPE, predicciones con intervalos de confianza y tendencia de los últimos 3 meses. Las respuestas se cachean en `session_state` para evitar llamadas repetidas.

El prompt está adaptado al rol:
- **Admin / Analista** — tono técnico; acepta preguntas sobre AIC, MAPE, walk-forward, parámetros del modelo o comparativa de algoritmos. El asistente conoce SARIMA, Prophet, Random Forest, XGBoost y Regresión Lineal.
- **Gerente** — tono accionable; orientado a recomendaciones de compra e interpretación de tendencias. Para comparar modelos, se redirige a la página **Comparativa ML**.

Requiere `GENAI_API_KEY` en `secrets.toml`. Si la clave no está configurada, el tab muestra un aviso en lugar de fallar.

---

## Seguridad

- Contraseñas gestionadas via `st.secrets` — nunca en el código fuente
- Timeout de sesión configurable (30 min por defecto)
- Verificación de credenciales: texto plano o hash SHA-256
- No se persiste información sensible en `session_state`

---

## .gitignore (entradas clave)

```
.streamlit/secrets.toml
*.pkl
*.xlsx
*.xls
*.csv
__pycache__/
.env
```

---

## Changelog

### 2026-03-28 (v9)
- **feat**: Dark premium UI aplicada a toda la aplicación — tema oscuro consistente (`#080D18` bg, `#20C997` teal, `#F59E0B` amber) en las cuatro páginas.
- **feat**: Nuevo módulo `styles.py` — CSS global centralizado con helpers `kpi_card()`, `section_header()` y `apply_chart_theme()`. Evita duplicar estilos en cada página; se inyecta una sola vez vía `show_header()`.
- **feat**: `.streamlit/config.toml` añadido — establece el tema base de Streamlit para que widgets nativos (botones, sliders, tabs) hereden el dark theme sin CSS extra.
- **feat**: `auth_system.py` rediseñado — login premium con tarjeta oscura y borde tricolor, header con logo invertido y divider, sidebar user-card con badge de rol y countdown de sesión.
- **feat**: `app_principal.py` — tarjetas de funcionalidades migradas de `st.info/success/warning` a componentes HTML `feature-card` (blue / green / amber) con tipografía Exo 2 + Barlow (Google Fonts).
- **feat**: `pages/2_Dashboard.py` — KPIs migrados a `kpi_card()` custom, todos los charts a `apply_chart_theme()` con paleta teal/amber; footer HTML premium.
- **feat**: `pages/1_Entrenamiento.py` y `pages/3_Comparativa_ML.py` — CSS legacy eliminado, heredan el global; `.winner-box` y demás clases movidas a `styles.py`.

### 2026-03-27 (v8)
- **feat**: Nueva página **🏆 Comparativa ML** (`pages/3_Comparativa_ML.py`) — reemplaza a Prophet vs SARIMA. Enfrenta hasta 5 modelos (SARIMA, Prophet, Regresión Lineal, Random Forest, XGBoost) sobre el mismo histórico mensual del Tiggo 2. Métricas: MAE, RMSE, MAPE y R². Incluye feature engineering con lag features (1,2,3,6,12 meses), medias móviles y features de calendario. Gráficas de predicciones, errores por mes e importancia de features para modelos ML. Botón de descarga CSV del período de test.
- **feat**: Landing page (`app_principal.py`) actualizada con tarjeta de acceso a Comparativa ML.
- **feat**: Asistente IA del Dashboard actualizado — los prompts de admin/analista ahora incluyen SARIMA, Prophet, Random Forest, XGBoost y Regresión Lineal. Se añade orientación hacia Comparativa ML en ambos roles.
- **chore**: Añadido `xgboost` a `requirements.txt`.

### 2026-03-25 (v7)
- **feat**: Búsqueda de hiperparámetros SARIMA migrada de grid search exhaustivo a **Optuna TPE** (80 trials bayesianos vs 384 combinaciones fijas, ~4× más rápido con igual calidad). La función `perform_grid_search` fue reemplazada por `perform_optuna_search`.
- **feat**: UI de resultados de búsqueda mejorada — expander *📊 Detalle de la búsqueda Optuna* con métricas en 3 columnas (trials evaluados, válidos, descartados) y explicación clara de por qué se descarta cada trial (predicciones negativas, fuera de rango, error numérico).
- **fix**: Corregido error `MS is not supported as period frequency` (`to_timestamp('MS')` → `to_timestamp()`).
- **chore**: Añadido `optuna` a `requirements.txt`.

### 2026-03-25 (v6)
- **feat**: Nueva página **⚔️ Prophet vs SARIMA** — comparación de ambos modelos sobre el mismo histórico con métricas MAE, RMSE, MAPE y tiempo de entrenamiento. Gráficas de predicciones, errores por mes y descomposición Prophet.
- **feat**: Landing page actualizada con tarjeta de acceso a Prophet vs SARIMA.
- **chore**: Añadido `prophet` a `requirements.txt`.

### 2026-03-25 (v5)
- **feat**: Limpieza automática de datos integrada en Tab 1 — elimina duplicados por `CHASIS` y filas con `MODELO3` nulo al cargar el Excel.
- **feat**: Grid search ampliado de 45 a 192 combinaciones — incluye `d∈{0,1}` y `P∈{0,1}`.
- **fix**: Criterio de selección del modelo cambiado de AIC mínimo a **MAPE mínimo** sobre el set de test.
- **fix**: Variables exógenas simplificadas a solo `ventas_otros`.
- **fix**: Dashboard Grid Search actualizado para mostrar y ordenar por MAPE.

### 2026-03-23 (v4)
- **feat**: Nueva pestaña **🤖 Asistente IA** en el Dashboard — chat sobre el modelo SARIMA entrenado, powered by Google Gemini (`gemini-2.5-flash`). Disponible para admin, analista y gerente. Requiere `GENAI_API_KEY` en `secrets.toml`.
- **chore**: Añadido `google-genai` a `requirements.txt`.

### 2026-03-23 (v3)
- **feat**: Nueva pestaña **🏪 Concesionarios** en el Dashboard — análisis de ventas CHERY por concesionario con KPIs, barras horizontales por ciudad, evolución mensual y ranking ABC.

### 2026-03-23 (v2)
- **feat**: Nueva pestaña **🎓 Preparar Datos** — pipeline académico paso a paso con descarga del `.xlsx` de entrenamiento.
- **feat**: Parámetro **Fecha fin de datos** en la pestaña Entrenamiento.

### 2026-03-23 (v1)
- **feat**: Detección automática de tipo de hoja al subir archivos — `Hoja1` → ventas, `Stock Actual` → stock.
- **fix**: Serialización JSON del test ADF corregida (`numpy.bool_` → `Python bool`).
- **fix**: Modelo `.pkl` comprimido con gzip antes de subir a Supabase Storage (resuelve error 413).

---

## Licencia

Uso interno. No distribuir públicamente.
