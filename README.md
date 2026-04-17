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
core/                         ← Paquete Python de utilidades
├── __init__.py
├── auth_system.py            ← Autenticación, sesiones y show_header()
├── supabase_io.py            ← Capa de I/O centralizada (Supabase Storage)
├── utils_validacion.py       ← Validación de datos de entrada
├── logger.py                 ← Logging centralizado (consola + archivo rotativo)
└── styles.py                 ← CSS global y helpers de componentes
tests/
└── test_validacion.py        ← 17 tests unitarios de utils_validacion
data/                         ← Datos locales — gitignored, nunca en el repo
├── raw/                      ← Excel histórico de ventas (fuente)
├── processed/                ← Datasets transformados (veh_ml_features.xlsx)
├── monthly/                  ← Ventas mensuales reales para actualización
└── artifacts/                ← Artefactos del modelo generados localmente
docs/
├── assets/                   ← Mockups e imágenes de desarrollo (gitignored)
├── 01_introduccion.md
├── 02_arquitectura.md
├── 03_guia_usuario.md
├── 04_modelos_ml.md          ← Documentación técnica de modelos ML
└── 05_despliegue.md
requirements.txt
.streamlit/
├── secrets.toml              ← Credenciales reales (NO en el repo)
└── secrets.toml.example      ← Plantilla (sí en el repo)
```

> No hay carpetas locales de datos persistentes en producción. Todos los artefactos del modelo se guardan y leen desde **Supabase Storage** (`bucket: modelos-ml`). La carpeta `data/` es solo para trabajo local.

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
    llm_cache.json                  ← Caché de respuestas Gemini (persistente por run)
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
| Fecha inicio | `2024-01-01` | Inicio de la **ventana de entrenamiento** — ver nota abajo |
| Fecha fin de datos | hoy | Límite superior del histórico. Se combina con «Eliminar mes actual»: gana el corte más conservador |
| Horizonte | `6` meses | Meses a predecir (3–12) |
| Máx. ventas | `100` unid./mes | Límite superior de predicciones válidas |
| Excluir mes actual | `true` | Eliminar el mes en curso (datos incompletos) |

### Ventana de entrenamiento

SARIMAX ajusta sus coeficientes al nivel medio y la estacionalidad del período que le muestras. Si el mercado cambió de régimen (nuevo precio, nueva competencia, quiebre post-pandemia), incluir datos del régimen anterior introduce **sesgo sistemático** — el modelo subestima o sobreestima de forma persistente.

**Regla práctica:** usa al menos **36 meses** (3 ciclos estacionales completos) del período que mejor represente el comportamiento actual.

| Situación | Fecha de inicio recomendada |
|-----------|----------------------------|
| Mercado estable | Máximo histórico disponible |
| Recuperación post-pandemia | 2021-01-01 |
| Nuevo nivel de demanda desde 2024 | **2024-01-01** ← caso TIGGO 2 |
| Lanzamiento de versión nueva | Fecha del lanzamiento |

La app muestra un **expander de ayuda** con el cálculo de meses en la ventana seleccionada y alertas:
- `< 36 meses` → ⚠️ advertencia (riesgo de coeficientes estacionales inestables)
- `36–48 meses` → ℹ️ info (ventana aceptable)
- `> 48 meses` → ✅ success (ventana robusta)

### Diagnóstico de MAPE alto

La pestaña Entrenamiento incluye una tabla de diagnóstico que se expande automáticamente cuando el MAPE supera el 20%:

| # | Causa probable | Señal | Solución |
|---|---------------|-------|----------|
| 1 | Outliers / errores de datos | Error en meses puntuales, no sistemático | Revisar el histórico y corregir datos |
| 2 | Horizonte demasiado largo | Error crece monotónicamente | Reducir el horizonte de predicción |
| 3 | Variable exógena incorrecta | Correlación < 0.3 con la serie objetivo | Revisar `ventas_otros` o eliminarla |
| 4 | Datos insuficientes | < 36 meses en la ventana | Ampliar la ventana de entrenamiento |
| 5 | **Quiebre estructural** | Sesgo negativo persistente varios meses | Mover «Fecha inicio» a la fecha del quiebre |

### Modelo SARIMA

- **Algoritmo**: SARIMAX con variable exógena (ventas de otros modelos de la misma marca)
- **Búsqueda de hiperparámetros**: [Optuna](https://optuna.org/) con sampler **TPE (Tree-structured Parzen Estimator)** — 80 trials bayesianos sobre el espacio `p ∈ {0–3}`, `d ∈ {0–1}`, `q ∈ {0–3}`, `P ∈ {0–1}`, `D ∈ {0–1}`, `Q ∈ {0–2}`, `m=12` (~4× más rápido que el grid search exhaustivo de 384 combinaciones, con igual o mejor calidad)
- **Criterio**: MAPE mínimo sobre el conjunto de test; se descartan predicciones fuera del rango `[0, max_ventas]`
- **Trials descartados**: predicciones negativas, superiores al límite configurado, errores numéricos de convergencia, o combinaciones `d=1 AND D=1` (sobre-diferenciación)
- **Variable exógena**: ventas mensuales de los demás modelos de la misma marca (`ventas_otros`). En el horizonte de predicción se usa la media móvil de los últimos 6 meses como valor asumido; el usuario ve un aviso antes del forecast
- **Intervalos de confianza**: 95% en todos los puntos del forecast
- **Walk-forward**: valida hasta los últimos **12 meses**, con mínimo igual al horizonte de predicción configurado

### Flujo de aprobación

La pestaña **Comparación** muestra métricas lado a lado con el modelo de producción (MAPE, AIC, próximo mes). Al hacer clic en **Aprobar** se actualiza `latest.txt` en Supabase y el Dashboard refleja el cambio inmediatamente. Los runs no aprobados quedan en el historial sin afectar producción.

---

## App 3 — Comparativa ML (`pages/3_Comparativa_ML.py`)

Página de comparación que enfrenta hasta **5 modelos** sobre el mismo histórico mensual de ventas del Tiggo 2. Accesible solo para `admin` y `analyst`.

### Modelos disponibles

| Modelo | Tipo | Enfoque |
|--------|------|---------|
| **SARIMA** | Serie de tiempo | Parámetros (p,d,q)(P,D,Q)₁₂ configurables manualmente |
| **Prophet** | Serie de tiempo | Estacionalidad multiplicativa anual + festivos de Perú (PE) |
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

### Publicar modelo ganador en producción

Al finalizar la comparativa aparece una sección **Publicar en producción** que:
- Muestra el modelo ganador y su MAPE
- Lanza una advertencia si el MAPE supera el 20%
- Si la fuente de datos fue un run de Supabase, permite activarlo como modelo de producción con un clic (actualiza `latest.txt`)
- Si la fuente fue un Excel manual, informa al usuario que debe entrenar un modelo desde la pestaña Entrenamiento

---

## App 2 — Dashboard (`pages/2_Dashboard.py`)

### Selector de versión

La barra lateral lista todos los runs disponibles ordenados por fecha. El run activo (apuntado por `latest.txt`) aparece con 🟢. Los históricos con 🔵 pueden seleccionarse sin alterar producción.

### Alertas de calidad del modelo

El KPI de MAPE en los tabs **Dashboard** y **Predicciones** cambia de color según el umbral:

| MAPE | Color | Mensaje |
|------|-------|---------|
| ≤ 10% | verde | Sin alerta — modelo de alta precisión |
| 10–20% | ámbar | Advertencia — precisión aceptable, monitorear |
| > 20% | rojo | Error — modelo de baja fiabilidad, reentrenar |

### Tab 🏪 Concesionarios (admin / analista / gerente)

Contiene el uploader de Excel con validación robusta: columnas requeridas marcadas como ❌ (bloqueante) o ⚠️ (advertencia no bloqueante), mensaje de error específico por columna ausente, normalización automática de nombres de columna y descarte de filas de cabecera extra.

- Columnas de fecha aceptadas: `FECHA_VENTA`, `FECHA-VENTA`, `FECHA VENTA`
- Columnas de modelo aceptadas: `MODELO2`, `MODELO3`, `MODELO`
- Columna de concesionario: `DET_CC` (prioridad) o `AGE`
- Columna de ciudad: `AGE` o `CIUDAD`/`REGION`
- Si la primera fila contiene descripciones (todo texto), se descarta automáticamente

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

Chat sobre el modelo entrenado, alimentado con Gemini (`gemini-2.5-flash`). El contexto que recibe el LLM incluye parámetros SARIMA, AIC/BIC, MAPE, predicciones con intervalos de confianza y tendencia de los últimos 3 meses. Las respuestas se cachean en `session_state` y se **persisten en Supabase** (`<run>/llm_cache.json`) para sobrevivir recargas de página. El caché se invalida automáticamente al cambiar de run en la barra lateral.

El prompt está adaptado al rol:
- **Admin / Analista** — tono técnico; acepta preguntas sobre AIC, MAPE, walk-forward, parámetros del modelo o comparativa de algoritmos
- **Gerente** — tono accionable; orientado a recomendaciones de compra e interpretación de tendencias

Requiere `GENAI_API_KEY` en `secrets.toml`. Si la clave no está configurada, el tab muestra un aviso en lugar de fallar.

---

## Módulo `core/`

Todos los módulos utilitarios residen en el paquete `core/` e importan entre sí con rutas relativas. Los archivos de `pages/` y `app_principal.py` los importan con la notación `from core.xxx import`.

| Módulo | Responsabilidad |
|--------|----------------|
| `core/auth_system.py` | Autenticación SHA-256, sesiones, timeout, RBAC, UI de login y header |
| `core/supabase_io.py` | Todas las operaciones de I/O con Supabase Storage (upload/download, runs, log) |
| `core/utils_validacion.py` | Validación de calidad del DataFrame antes de entrenar |
| `core/logger.py` | Logger centralizado — escribe a consola y a `logs/app.log` (rotativo 2 MB × 3) |
| `core/styles.py` | CSS global dark premium, helpers `kpi_card()`, `section_header()`, `apply_chart_theme()` |

---

## Seguridad

- Contraseñas gestionadas via `st.secrets` — nunca en el código fuente
- Timeout de sesión configurable (30 min por defecto)
- Verificación de credenciales: texto plano o hash SHA-256
- No se persiste información sensible en `session_state`

---

## Tests

```bash
pytest tests/ -v
```

Los tests de `tests/test_validacion.py` no dependen de Streamlit; validan la lógica pura de `validate_dataframe` y `get_dataset_summary` en 6 clases / 17 casos:

| Clase | Qué cubre |
|-------|-----------|
| `TestColumnas` | Columnas requeridas presentes / ausentes |
| `TestPeriodoTemporal` | Umbral mínimo de 36 meses |
| `TestFechasInvalidas` | Fechas inválidas por encima/debajo del umbral |
| `TestDatosFaltantes` | Warnings de nulos |
| `TestGetDatasetSummary` | Claves y conteos del resumen |
| `TestCasosFrontera` | 1 fila, constantes de configuración |

---

## .gitignore (entradas clave)

```
.streamlit/secrets.toml     ← credenciales
data/                       ← todos los datos locales (xlsx, pkl, csv, json)
*.pkl  *.xlsx  *.xls  *.csv ← captura por extensión (backup)
mockup_*.py  mockup_*.png   ← artefactos de desarrollo
docs/assets/                ← imágenes de mockup
logs/  *.log                ← logs generados en runtime
__pycache__/  venv/  .env   ← estándar Python
```

---

## Changelog

### 2026-04-16 (v14)
- **refactor**: Profesionalización de la estructura de carpetas — módulos Python extraídos a paquete `core/` (`auth_system`, `logger`, `styles`, `supabase_io`, `utils_validacion`).
- **refactor**: Importaciones actualizadas en todos los archivos: `from core.xxx import` / `import core.xxx as`. Importaciones internas dentro de `core/` convertidas a relativas (`.module`).
- **fix**: `core/logger.py` — corregido `_LOGS_DIR` para que los logs se escriban en la raíz del proyecto (`logs/`) y no dentro de `core/logs/`.
- **chore**: Datos locales reorganizados en `data/{raw,processed,monthly,artifacts}/` (gitignored). `datos_reales/` → `data/monthly/`, `data dashboard/` → `data/artifacts/`.
- **chore**: Mockup de sprint movido a `docs/assets/` (gitignored). `.gitignore` actualizado con los nuevos patrones de ruta.

### 2026-04-16 (v13)
- **feat**: Constante `TRAINING_DEFAULT_START = date(2024, 1, 1)` — ventana de entrenamiento por defecto cambiada de 2021-01-01 a 2024-01-01 para aislar el régimen de demanda actual del TIGGO 2 (~65 uds/mes) del régimen anterior (~31 uds/mes).
- **feat**: Widget `fecha_inicio` con `help` extenso que explica el concepto de quiebre estructural y la regla de 36 meses mínimos.
- **feat**: Nuevo expander "¿Cómo elegir la ventana de entrenamiento?" con tabla de casos de uso, cálculo dinámico de meses en la ventana y alertas contextuales (< 36 m ⚠️, 36–48 m ℹ️, ≥ 48 m ✅).
- **feat**: Tabla de diagnóstico de MAPE — fila 5 actualizada con el caso concreto del TIGGO 2 como ejemplo de quiebre estructural (MAPE 42%).
- **feat**: Campo `meses_ventana` añadido al dict de métricas guardado en Supabase para auditoría por run.
- **data**: `veh_ml_features.xlsx` actualizado a 30.039 filas (hasta 2026-03-31) con datos reales de Feb y Mar 2026 (64 y 66 unidades TIGGO 2 confirmadas).

### 2026-04-15 (v12)
- **fix**: `docs/04_modelos_ml.md` corregido a `country_name='PE'` (festivos peruanos) alineando con la implementación real de Prophet.
- **refactor**: Magic numbers extraídos a constantes nombradas en `1_Entrenamiento.py` y `3_Comparativa_ML.py`.
- **fix**: `warnings.filterwarnings('ignore')` global reemplazado por supresión acotada solo a módulos `statsmodels`.
- **fix**: Todos los `except Exception: pass` en `supabase_io.py` ahora registran `log.debug(...)` con contexto.
- **feat**: Validación anticipada de `max_ventas` antes de lanzar Optuna — error bloqueante si el límite es menor que el pico histórico.
- **fix**: Traceback raw reemplazado por mensaje amigable + expander colapsado con detalle técnico.
- **feat**: Límite de 500 caracteres (`max_chars=500`) en inputs del Asistente IA.
- **docs**: Docstrings completos en `run_adf_test`, `train_sarima_model`, `perform_optuna_search` y `perform_walk_forward`.

### 2026-04-15 (v11)
- **feat**: Nuevo módulo `logger.py` — logging centralizado a consola + archivo rotativo (`logs/app.log`, 2 MB × 3 backups). Integrado en `auth_system.py` y `supabase_io.py`.
- **feat**: Nueva suite de tests `tests/test_validacion.py` — 17 tests en 6 clases sin dependencia de Streamlit.
- **refactor**: Type hints completos en `auth_system.py`, `supabase_io.py` y `utils_validacion.py`.
- **feat**: Diagnóstico de MAPE > 20% con tabla de 5 causas y soluciones, expandido automáticamente.
- **docs**: `docs/04_modelos_ml.md` ampliado con restricciones del espacio de búsqueda Optuna y limitaciones de la proyección de variable exógena.

### 2026-04-04 (v10)
- **fix**: Restricción `d=1 AND D=1` en Optuna para evitar sobre-diferenciación.
- **feat**: Walk-forward validation extendido a 12 meses (antes 6).
- **feat**: Aviso informativo de asunción de variable exógena antes del forecast.
- **feat**: Alertas dinámicas de MAPE en Dashboard y Predicciones (rojo / ámbar / verde).
- **feat**: Sección "Publicar modelo ganador en producción" en Comparativa ML.
- **feat**: Caché de respuestas Gemini persistido en Supabase (`<run>/llm_cache.json`).
- **feat**: Uploader de concesionarios movido del sidebar al tab 🏪, con validación robusta por columna.

### 2026-03-28 (v9)
- **feat**: Dark premium UI aplicada a toda la aplicación (`#080D18` bg, `#20C997` teal, `#F59E0B` amber).
- **feat**: Nuevo módulo `styles.py` con CSS global centralizado y helpers `kpi_card()`, `section_header()`, `apply_chart_theme()`.
- **feat**: `.streamlit/config.toml` añadido para dark theme nativo en widgets Streamlit.
- **feat**: Login premium con tarjeta oscura, borde tricolor, sidebar user-card con badge de rol y countdown de sesión.

### 2026-03-27 (v8)
- **feat**: Nueva página **🏆 Comparativa ML** — enfrenta SARIMA, Prophet, Regresión Lineal, Random Forest y XGBoost. Feature engineering con lag features y calendario. Gráficas de predicciones, errores e importancia de features. Descarga CSV.
- **chore**: `xgboost` añadido a `requirements.txt`.

### 2026-03-25 (v7)
- **feat**: Búsqueda de hiperparámetros migrada de grid search exhaustivo a **Optuna TPE** (80 trials vs 384 combinaciones, ~4× más rápido).
- **fix**: Error `MS is not supported as period frequency` corregido.
- **chore**: `optuna` añadido a `requirements.txt`.

### 2026-03-25 (v6)
- **feat**: Nueva página **⚔️ Prophet vs SARIMA** (antecesora de Comparativa ML).
- **chore**: `prophet` añadido a `requirements.txt`.

### 2026-03-25 (v5)
- **feat**: Limpieza automática al cargar — elimina duplicados por `CHASIS` y filas con `MODELO3` nulo.
- **feat**: Grid search ampliado a 192 combinaciones con `d∈{0,1}` y `P∈{0,1}`.
- **fix**: Criterio de selección cambiado de AIC mínimo a MAPE mínimo.

### 2026-03-23 (v4)
- **feat**: Tab **🤖 Asistente IA** en el Dashboard — chat SARIMA con Gemini (`gemini-2.5-flash`).
- **chore**: `google-genai` añadido a `requirements.txt`.

### 2026-03-23 (v3)
- **feat**: Tab **🏪 Concesionarios** — análisis de ventas CHERY por concesionario con KPIs, barras horizontales y ranking ABC.

### 2026-03-23 (v2)
- **feat**: Tab **🎓 Preparar Datos** — pipeline académico paso a paso con descarga del `.xlsx`.
- **feat**: Parámetro **Fecha fin de datos** en pestaña Entrenamiento.

### 2026-03-23 (v1)
- **feat**: Detección automática de tipo de hoja (`Hoja1` → ventas, `Stock Actual` → stock).
- **fix**: Serialización JSON del test ADF (`numpy.bool_` → `Python bool`).
- **fix**: Modelo `.pkl` comprimido con gzip antes de subir a Supabase (resuelve error 413).

---

## Licencia

Uso interno. No distribuir públicamente.
