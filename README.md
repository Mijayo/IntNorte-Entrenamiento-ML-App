# Sistema TIGGO 2 — Predicción de Ventas con SARIMA y Prophet

Sistema multipage en Streamlit para entrenar modelos SARIMA, comparar con Prophet y visualizar predicciones de demanda de vehículos. Desplegado en **Streamlit Cloud** con almacenamiento persistente en **Supabase Storage** y control de acceso por roles.

---

## Arquitectura

```
app_principal.py              ← Entry point (autenticación + página de inicio)
pages/
├── 1_Entrenamiento.py        ← Entrenamiento SARIMA (Admin / Analista)
├── 2_Dashboard.py            ← Dashboard de negocio (todos los roles)
└── 3_Prophet_vs_SARIMA.py    ← Comparación Prophet vs SARIMA (Admin / Analista)
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
matplotlib, plotly, pillow, openpyxl, supabase, google-genai, prophet
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

| Rol | Entrenamiento | Prophet vs SARIMA | Tabs del Dashboard |
|-----|:-------------:|:-----------------:|---------------------|
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
5. [Entrenamiento]  Grid search → mejores parámetros SARIMA (criterio: MAPE mínimo)
6. [Entrenamiento]  Walk-forward validation sobre los últimos N meses
6. [Entrenamiento]  Modelo final + forecast generado
7. [Entrenamiento]  Artefactos subidos automáticamente a Supabase Storage
8. [Entrenamiento]  Comparación con el modelo de producción actual
9. [Entrenamiento]  Clic en "Aprobar" → latest.txt actualizado → Dashboard activo
10. [Dashboard]     Carga el modelo activo automáticamente al arrancar
11. [Dashboard]     Barra lateral permite cambiar entre cualquier run histórico
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
| 4 | 🤖 Entrenamiento | Configuración, grid search, walk-forward, guardado en Supabase |
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
- **Espacio de búsqueda**: `p ∈ {0,1,2,3}`, `d ∈ {0,1}`, `q ∈ {0,1,2,3}`, estacional `(P∈{0,1}, D∈{0,1}, Q∈{0,1,2}, m=12)` → 192 combinaciones
- **Criterio**: MAPE mínimo sobre el conjunto de test con predicciones dentro del rango configurado
- **Variable exógena**: ventas mensuales de los demás modelos de la misma marca (`ventas_otros`)
- **Intervalos de confianza**: 95% en todos los puntos del forecast

### Flujo de aprobación

La pestaña **Comparación** muestra métricas lado a lado con el modelo de producción (MAPE, AIC, próximo mes). Al hacer clic en **Aprobar** se actualiza `latest.txt` en Supabase y el Dashboard refleja el cambio inmediatamente. Los runs no aprobados quedan en el historial sin afectar producción.

---

## App 3 — Prophet vs SARIMA (`pages/3_Prophet_vs_SARIMA.py`)

Página de comparación académica/técnica que enfrenta Prophet (Meta) contra SARIMA sobre el mismo histórico de ventas. Accesible solo para `admin` y `analyst`.

### Flujo de la comparación

| Paso | Descripción |
|------|-------------|
| 1. Fuente de datos | Carga el histórico desde un run guardado en Supabase **o** sube un Excel propio |
| 2. Configuración | Meses de test (hold-out), festivos MX para Prophet, parámetros SARIMA manuales |
| 3. Entrenar | Entrena ambos modelos y mide el tiempo de ejecución |
| 4. Resultados | Tabla de métricas, gráficas, descomposición de Prophet y explicación didáctica |

### Métricas comparadas

| Métrica | Descripción |
|---------|-------------|
| MAE | Error absoluto medio (unidades) |
| RMSE | Raíz del error cuadrático medio |
| MAPE (%) | Error porcentual medio — criterio principal de desempate |
| AIC | Solo disponible para SARIMA |
| Tiempo (s) | Tiempo de entrenamiento de cada modelo |

### Configuración de Prophet

- **Estacionalidad**: anual multiplicativa (apta para series con tendencia creciente)
- **Festivos**: `add_country_holidays('MX')` — Día de Muertos, Navidad, Año Nuevo, etc.
- **Frecuencia**: mensual (`MS` — inicio de mes)

### Gráficas de resultados

1. **Predicciones en test** — histórico train + real test + SARIMA + Prophet en el mismo eje
2. **Error absoluto por mes** — barras agrupadas para comparar mes a mes
3. **Descomposición Prophet** — tendencia + estacionalidad anual en ejes separados

### ¿Por qué Prophet?

| Característica | SARIMA | Prophet |
|---|---|---|
| Estacionalidad múltiple | Solo una (mensual) | Anual + mensual + eventos |
| Festivos/eventos | Manual, complejo | Nativo (`holidays=`) |
| Datos faltantes | Relleno manual | Manejados automáticamente |
| Cambios de tendencia | Difícil | Detecta *changepoints* automáticamente |
| Calibración | Grid search de 300+ combinaciones | Pocos hiperparámetros intuitivos |
| Interpretabilidad | Difícil de explicar | Descompone: tendencia + estacionalidad |

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
- **Admin / Analista** — tono técnico; acepta preguntas sobre AIC, MAPE, walk-forward o parámetros del modelo.
- **Gerente** — tono accionable; orientado a recomendaciones de compra e interpretación de tendencias.

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

### 2026-03-25 (v6)
- **feat**: Nueva página **⚔️ Prophet vs SARIMA** (`pages/3_Prophet_vs_SARIMA.py`) — compara ambos modelos sobre el mismo histórico de ventas con métricas MAE, RMSE, MAPE y tiempo de entrenamiento. Carga datos desde cualquier run de Supabase o desde un Excel subido manualmente. Incluye gráficas de predicciones, errores por mes, descomposición Prophet (tendencia + estacionalidad anual) y explicación didáctica del resultado.
- **feat**: Landing page actualizada con tarjeta de acceso a Prophet vs SARIMA.
- **chore**: Añadido `prophet` a `requirements.txt`.

### 2026-03-25 (v5)
- **feat**: Limpieza automática de datos integrada en Tab 1 — elimina duplicados por `CHASIS` (conserva el registro más reciente) y filas con `MODELO3` nulo al cargar el Excel. Muestra conteo de filas eliminadas.
- **feat**: Grid search ampliado de 45 a 192 combinaciones — incluye `d∈{0,1}` y `P∈{0,1}` (antes fijos a 1).
- **fix**: Criterio de selección del modelo cambiado de AIC mínimo a **MAPE mínimo** sobre el set de test, alineado con el objetivo predictivo.
- **fix**: Variables exógenas simplificadas a solo `ventas_otros` — eliminadas `tendencia` y `mes` por redundancia con los componentes internos de SARIMA.
- **fix**: Dashboard Grid Search actualizado para mostrar y ordenar por MAPE (antes ordenaba por AIC).
- **chore**: UI Tab 1 actualizada — botón "Procesar" y texto en singular para el caso de un único archivo.

### 2026-03-23 (v4)
- **feat**: Nueva pestaña **🤖 Asistente IA** en el Dashboard — chat sobre el modelo SARIMA entrenado, powered by Google Gemini (`gemini-2.5-flash`). Disponible para admin, analista y gerente. El prompt se adapta al rol: técnico para admin/analista, accionable para gerente. Las respuestas se cachean en `session_state`. Requiere `GENAI_API_KEY` en `secrets.toml`.
- **chore**: Añadido `google-genai` a `requirements.txt` y `GENAI_API_KEY` a `secrets.toml.example`.

### 2026-03-23 (v3)
- **feat**: Nueva pestaña **🏪 Concesionarios** en el Dashboard — análisis de ventas CHERY por concesionario con KPIs, barras horizontales por ciudad, evolución mensual y ranking ABC. Accesible para admin, analista y gerente. Los datos se cargan desde la barra lateral (expander *📂 Datos de Concesionarios*) y se normalizan automáticamente (columnas de fecha, modelo y concesionario con alias múltiples).

### 2026-03-23 (v2)
- **feat**: Nueva pestaña **🎓 Preparar Datos** — pipeline académico paso a paso con descarga del `.xlsx` de entrenamiento.
- **feat**: Parámetro **Fecha fin de datos** en la pestaña Entrenamiento — permite acotar el histórico por una fecha máxima explícita; se combina con «Eliminar mes actual» tomando el corte más conservador. Se persiste en `metricas` (Supabase).

### 2026-03-23 (v1)
- **feat**: Detección automática de tipo de hoja al subir archivos — `Hoja1` → ventas, `Stock Actual` → stock. Permite cargar ambos tipos en una misma subida sin configuración adicional.
- **fix**: Serialización JSON del test ADF corregida (`numpy.bool_` → `Python bool`) para evitar error al guardar métricas en Supabase.
- **fix**: Modelo `.pkl` comprimido con gzip antes de subir a Supabase Storage, resolviendo el error 413 (payload demasiado grande) en modelos grandes.

---

## Licencia

Uso interno. No distribuir públicamente.
