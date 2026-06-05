# Página 1 — Entrenamiento de Modelos SARIMA

**Archivo:** `pages/1_Entrenamiento.py`  
**Acceso:** `admin`, `analyst` (permiso `entrenar_modelos`)  
**Icono:** 🤖

---

## Propósito

Permite cargar el histórico de ventas del Chery Tiggo 2, validarlo, entender cómo se construye la serie temporal y lanzar el entrenamiento del modelo SARIMA con búsqueda bayesiana de hiperparámetros (Optuna TPE). El modelo entrenado se guarda en Supabase y queda disponible para activarse en el Dashboard.

---

## Parámetros de configuración (constantes)

| Constante | Valor | Descripción |
|-----------|------:|-------------|
| `OPTUNA_N_TRIALS` | 80 | Trials bayesianos TPE (≈ 4× más rápido que un grid exhaustivo de 384) |
| `SARIMA_SEASONAL_PERIOD` | 12 | Período estacional mensual |
| `WALK_FORWARD_MONTHS` | 12 | Ventana máxima de validación walk-forward |
| `EXOG_ROLLING_WINDOW` | 6 | Meses usados para proyectar `ventas_otros` en el horizonte |
| `TRAINING_DEFAULT_START` | 2022-01-01 | Fecha de inicio por defecto de la ventana de entrenamiento |
| `TRAINING_DEFAULT_END` | 2026-03-31 | Fecha fin por defecto (último mes con datos completos) |

---

## Pestañas (flujo recomendado en orden)

### 📤 Pestaña 1 — Cargar Datos

Dos modos seleccionables mediante radio button:

#### Modo A — Datos precargados *(recomendado)*

Carga `data/processed/veh_ml_features.xlsx` desde caché local (Ene 2017 – Mar 2026, ~30 039 registros).

1. Deja seleccionada la opción **📦 Datos precargados**.
2. Haz clic en **✅ Cargar**.

La primera carga lee el disco; las siguientes usan caché en memoria (`@st.cache_data`).

#### Modo B — Subir nuevo Excel

Para actualizar el histórico con datos más recientes:

1. Selecciona uno o varios archivos `.xlsx` (se unifican automáticamente).
2. El sistema detecta el tipo de hoja:
   - `Hoja1` → archivo de ventas
   - `Stock Actual` → archivo de stock
3. Pulsa **🔄 Procesar**.

#### Limpieza automática (ambos modos)

- Elimina filas con `MODELO3` vacío.
- Elimina duplicados de chasis, conservando el registro más reciente (`sort + drop_duplicates`).

---

### ✅ Pestaña 2 — Validación

Ejecuta automáticamente los checks de calidad:

| Check | ¿Qué verifica? |
|-------|----------------|
| Columnas requeridas | `FECHA-VENTA`, `MODELO3`, `MARCA` presentes |
| Fechas válidas | < 5% de fechas nulas o inválidas |
| Período mínimo | ≥ 36 meses de histórico |
| Datos faltantes | < 5% de nulos en columnas clave |
| Outliers | Meses con ventas > 3σ de la media |

- **Errores bloqueantes** (rojo): impiden continuar hasta corregirlos.
- **Advertencias** (amarillo): permiten continuar pero deben evaluarse.

Muestra además distribución temporal y mapa de datos faltantes.

---

### 🎓 Pestaña 3 — Preparar Datos *(académico)*

Pestaña informativa que muestra paso a paso la transformación del Excel bruto en la serie temporal mensual que alimenta SARIMA. **No entrena nada.**

| Paso | Operación |
|------|-----------|
| 1 | Datos brutos — muestra las primeras filas del Excel |
| 2 | Filtro por marca (p.ej. CHERY) |
| 3 | Filtro por modelo (p.ej. TIGGO 2) |
| 4 | Recorte por rango de fechas |
| 5 | Resample mensual (`resample('ME').size()`) — convierte filas individuales en conteos mensuales |
| 6 | Variable exógena — suma mensual de ventas de los otros modelos de la misma marca |

Incluye botón **📥 Descargar Excel de entrenamiento** con tres hojas: `Serie_SARIMA`, `Ventas_Mensuales`, `Comparativa`.

---

### 🤖 Pestaña 4 — Entrenamiento

**Parámetros configurables:**

| Parámetro | Por defecto | Descripción |
|-----------|:-----------:|-------------|
| Filtro Marca | CHERY | Filtra el Excel por marca |
| Filtro Modelo | TIGGO 2 | Modelo objetivo a predecir |
| Fecha inicio | 2022-01-01 | Límite inferior de la ventana de entrenamiento |
| Fecha fin datos | 2026-03-31 | Límite superior del histórico |
| Eliminar mes actual | ✅ Sí | Excluye el mes en curso (datos incompletos) |
| Horizonte (meses) | 6 | Meses hacia adelante a predecir |
| Límite máximo ventas/mes | 100 | Descarta combinaciones con predicciones fuera de rango |

**Expander "📅 ¿Cómo elegir la ventana de entrenamiento?"**

Explica el concepto de quiebre estructural (*structural break*) y ofrece una tabla de fechas de inicio recomendadas según la situación del mercado:

| Situación | Fecha inicio recomendada |
|-----------|--------------------------|
| Mercado estable | Máximo histórico disponible (2017+) |
| Recuperación post-pandemia | 2021-01-01 |
| Cobertura completa TIGGO 2 | **2022-01-01** ← caso actual |
| Nuevo régimen de demanda reciente | 2024-01-01 |

El mínimo estadístico de SARIMA es **36 meses** (3 ciclos estacionales completos). La UI advierte si la ventana es menor.

**Secuencia de entrenamiento al pulsar 🚀 Iniciar Entrenamiento:**

1. **Validación anticipada** — verifica que `max_ventas` supere el pico histórico del modelo; si no, detiene el proceso con mensaje explicativo.
2. **Preparar datos** — filtra y resamplea la serie mensual.
3. **Filtro de correlación exógena** — calcula Pearson r entre `ventas_tiggo2` y `ventas_otros`:
   - `|r| ≥ 0.3` → incluye la variable exógena.
   - `|r| < 0.3` → descarta la exógena (entrena SARIMA puro).
4. **Test ADF** (Dickey-Fuller Aumentado) — verifica estacionariedad de la serie. Si `p > 0.05`, se aplica diferenciación `d=1`.
5. **ACF/PACF** — genera los gráficos de autocorrelación (guardados en Supabase para el Dashboard).
6. **Búsqueda Optuna TPE** — 80 trials bayesianos. Espacio de búsqueda: `p∈{0-3}, d∈{0-1}, q∈{0-3}, P∈{0-1}, D∈{0-1}, Q∈{0-2}`. Se descarta `d=1 ∧ D=1` (sobre-diferenciación). Criterio: minimizar MAPE en el conjunto de test.
7. **Walk-forward validation** — reentrena el modelo para cada uno de los últimos 12 meses y predice un paso adelante. Calcula MAPE real y detecta drift.
8. **Modelo final** — entrena sobre el histórico completo con los mejores parámetros. Proyecta la variable exógena mediante tendencia lineal sobre los últimos `EXOG_ROLLING_WINDOW*2` meses.
9. **Guardar en Supabase** — almacena el modelo (pickle gzipado), predicciones, grid search, walk-forward, imágenes ACF/PACF y métricas.

**Diagnóstico automático de MAPE:**

| MAPE | Estado |
|------|--------|
| ≤ 10% | ✅ Excelente |
| 10–15% | ⚠️ Aceptable |
| > 15% | ❌ Alto — muestra tabla de causas probables y acciones recomendadas |

---

### 📊 Pestaña 5 — Comparación

Disponible solo después de completar un entrenamiento. Compara el nuevo modelo con el que está activo en producción:

| Métrica | Modelo actual | Nuevo | Δ |
|---------|:------------:|:-----:|---|
| MAPE Walk-Forward | — | — | — |
| AIC | — | — | — |
| Predicción próximo mes | — | — | — |
| Modelo (parámetros) | — | — | — |

Recomendación automática:
- ✅ **APROBAR** — mejora MAPE y AIC.
- ⚠️ **REVISAR** — mejora en alguna métrica pero no en todas.
- ❌ **NO APROBAR** — empeora ambas métricas.

Muestra además residuos del nuevo modelo (gráfico temporal + histograma) con métricas de media, desviación estándar y residuo máximo absoluto.

**Botón "✅ Aprobar y activar en Dashboard"** — marca el run como modelo activo. El Dashboard lo refleja de inmediato.

---

### 📋 Pestaña 6 — Historial

Consulta todos los entrenamientos pasados registrados en `training_log` de Supabase:

- KPIs: total de ejecuciones, último MAPE, mejor MAPE histórico.
- Tabla completa: fecha, usuario, modelo SARIMA seleccionado, AIC, MAPE WF, horizonte, trials válidos/total.
- Gráfico de evolución del MAPE en el tiempo.
- Botón **📥 Exportar historial CSV**.

---

## Funciones internas

| Función | Descripción |
|---------|-------------|
| `run_adf_test(series)` | Test de Dickey-Fuller Aumentado; devuelve dict con estadístico, p-valor y `is_stationary` |
| `train_sarima_model(ventas, exog_data, order, seasonal_order)` | Entrena un SARIMAX con `maxiter=200, method='lbfgs'` |
| `perform_optuna_search(...)` | Búsqueda bayesiana TPE con 80 trials; retorna `(best_params, best_aic, best_mape, trial_results, n_discarded)` |
| `perform_walk_forward(...)` | Rolling-origin validation; retorna lista de dicts con `{fecha, real, prediccion, error, error_pct}` |
| `plot_residuals(model_results)` | Genera gráfico Plotly con residuos en el tiempo + histograma |
| `_load_preloaded()` | Carga `veh_ml_features.xlsx` con `@st.cache_data` |
| `_clean_ventas_df(df)` | Limpia duplicados de chasis y nulos en `MODELO3` |
