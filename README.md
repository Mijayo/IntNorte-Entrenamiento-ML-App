# Sistema TIGGO 2 — Predicción de Ventas con ML

Sistema multipage en Streamlit para entrenar modelos SARIMA, comparar múltiples algoritmos de predicción (SARIMA, Prophet, Regresión Lineal, Random Forest, XGBoost) y visualizar predicciones de demanda del Chery Tiggo 2. Desplegado en **Streamlit Cloud** con almacenamiento persistente en **Supabase Storage**, base de datos relacional en **Supabase PostgreSQL** y autenticación via **Supabase Auth**.

---

## Documentación

| Documento | Contenido |
|-----------|-----------|
| [01 — Introducción](docs/01_introduccion.md) | Contexto de negocio, objetivos y alcance del proyecto |
| [02 — Arquitectura](docs/02_arquitectura.md) | Stack técnico, flujo de datos, módulos y decisiones de diseño |
| [03 — Guía de usuario](docs/03_guia_usuario.md) | Uso paso a paso de cada página de la aplicación |
| [04 — Modelos ML](docs/04_modelos_ml.md) | SARIMA, Prophet, Optuna TPE, walk-forward validation, métricas |
| [05 — Despliegue](docs/05_despliegue.md) | Streamlit Cloud, Supabase, configuración de secretos y tablas SQL |
| [06 — Conclusiones Iteración 1](docs/06_conclusiones_iteracion1.md) | Resultados, aprendizajes y decisiones de la primera iteración |
| [07 — Conclusiones Iteración 2](docs/07_conclusiones_iteracion2.md) | Resultados, aprendizajes y decisiones de la segunda iteración |

---

## Arquitectura

```
app_principal.py              ← Entry point (autenticación + página de inicio)
pages/
├── 1_Entrenamiento.py        ← Entrenamiento SARIMA (Admin / Analista)
├── 2_Comparativa_ML.py       ← Comparativa de 5 modelos ML (Admin / Analista)
├── 3_Dashboard.py            ← Dashboard de negocio (todos los roles)
├── 4_Concesionarios.py       ← Análisis histórico + predicciones por tienda (Admin / Analista / Manager)
├── 5_Proyeccion_Ingresos.py  ← Proyección financiera en USD (Admin / Analista / Financiero)
└── 6_Escalabilidad.py        ← Hoja de ruta multi-marca: portafolio, líneas de negocio, onboarding, LatAm (todos)
core/                         ← Paquete Python de utilidades
├── __init__.py
├── auth_system.py            ← Autenticación Supabase Auth + fallback local, sesiones, RBAC
├── supabase_io.py            ← Capa de I/O centralizada (Storage + PostgreSQL + Audit Log)
├── utils_validacion.py       ← Validación de datos de entrada
├── logger.py                 ← Logging centralizado (consola + archivo rotativo)
└── styles.py                 ← CSS global y helpers de componentes
tests/
└── test_validacion.py        ← 19 tests unitarios de utils_validacion
data/                         ← Datos locales — gitignored, nunca en el repo
├── raw/                      ← Excel histórico de ventas (fuente)
├── processed/                ← Datasets transformados (veh_ml_features.xlsx — incluye columna ventas_otros)
├── monthly/                  ← Ventas mensuales reales para actualización
└── artifacts/                ← Artefactos del modelo generados localmente
docs/
├── assets/                   ← Mockups e imágenes de desarrollo (gitignored)
├── 01_introduccion.md
├── 02_arquitectura.md
├── 03_guia_usuario.md
├── 04_modelos_ml.md
├── 05_despliegue.md
├── 06_conclusiones_iteracion1.md
└── 07_conclusiones_iteracion2.md
requirements.txt
.streamlit/
├── secrets.toml              ← Credenciales reales (NO en el repo)
└── secrets.toml.example      ← Plantilla (sí en el repo)
```

> No hay carpetas locales de datos persistentes en producción. Los artefactos del modelo se guardan en **Supabase Storage** (`bucket: modelos-ml`). Los metadatos de runs se almacenan en **Supabase PostgreSQL** (`tabla: training_runs`). La carpeta `data/` es solo para trabajo local.

---

## Supabase: Storage + PostgreSQL

### Bucket Storage (`modelos-ml`)

```
latest.txt                          ← Apunta al run de producción activo (backup)
training_log.json                   ← Historial de runs (backup; primario es la DB)
YYYYMMDD_HHMMSS/                    ← Una carpeta por run de entrenamiento
    metricas_mejoradas.json
    prediccion_total_mejorada.xlsx
    grid_search_results.xlsx
    walk_forward_validation.xlsx
    historico_total_mejorado.xlsx
    historico_exog.xlsx             ← Serie ventas_otros usada como exógena (si |Pearson r| ≥ 0.3)
    modelo_total_mejorado.pkl.gz
    acf_plot.png
    pacf_plot.png
    llm_cache.json                  ← Caché de respuestas Gemini (persistente por run)
```

### Tablas PostgreSQL

**`training_runs`** — Registro primario de todos los entrenamientos:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `run_name` | TEXT UNIQUE | Timestamp del run (`YYYYMMDD_HHMMSS`) |
| `usuario` | TEXT | Usuario que entrenó |
| `modelo` / `marca` | TEXT | Filtros usados |
| `horizonte` | INT | Meses de predicción |
| `aic` / `mape_wf` | NUMERIC | Métricas del modelo |
| `sarima_order` / `sarima_seasonal` | TEXT (JSON) | Parámetros SARIMA |
| `activo` | BOOLEAN | `TRUE` = modelo en producción |
| `created_at` | TIMESTAMPTZ | Fecha de inserción |

**`audit_log`** — Trazabilidad de acciones:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `usuario` | TEXT | Usuario que realizó la acción |
| `accion` | TEXT | `LOGIN`, `LOGOUT`, `APPROVE_MODEL`, `DELETE_RUN` |
| `run_name` | TEXT | Run afectado (nullable) |
| `detalle` | JSONB | Información adicional |
| `timestamp` | TIMESTAMPTZ | Fecha automática |

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
# Editar con URL, anon key de Supabase y configuración de usuarios
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

# Usuarios — el campo 'email' activa Supabase Auth; 'password' es el fallback local
[users.admin]
email    = "admin@tudominio.com"   # cuenta creada en Supabase Auth
password = "hash_sha256_opcional"  # fallback si Supabase Auth no está disponible
role     = "admin"
name     = "Administrador"
icon     = "👑"

[users.admin.permissions]
entrenar_modelos      = true
ver_metricas_tecnicas = true
ver_predicciones      = true
exportar              = true
gestionar_usuarios    = true
ver_grid_search       = true
ver_acf_pacf          = true
ver_ingresos          = true
# ... ver secrets.toml.example para la lista completa
```

### 2. Tablas Supabase (SQL Editor)

Ejecuta en **Supabase → SQL Editor**:

```sql
-- Tabla principal de entrenamientos
CREATE TABLE IF NOT EXISTS training_runs (
  id                        BIGSERIAL    PRIMARY KEY,
  run_name                  TEXT         NOT NULL UNIQUE,
  timestamp                 TIMESTAMPTZ,
  usuario                   TEXT,
  modelo                    TEXT,
  marca                     TEXT,
  fecha_inicio              TEXT,
  horizonte                 INT,
  max_ventas                INT,
  sarima_order              TEXT,
  sarima_seasonal           TEXT,
  aic                       NUMERIC(10,2),
  mape_wf                   NUMERIC(6,2),
  meses_datos               INT,
  combinaciones_validas     INT,
  combinaciones_descartadas INT,
  activo                    BOOLEAN      NOT NULL DEFAULT FALSE,
  created_at                TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_training_runs_activo
  ON training_runs (activo) WHERE activo = TRUE;

-- Tabla de audit log
CREATE TABLE IF NOT EXISTS audit_log (
  id        BIGSERIAL    PRIMARY KEY,
  timestamp TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  usuario   TEXT,
  accion    TEXT         NOT NULL,
  run_name  TEXT,
  detalle   JSONB
);
```

### 3. Supabase Auth — Crear usuarios

1. En Supabase → **Authentication → Users → Invite user** (o Add user).
2. Introduce el email que aparece en `secrets.toml` bajo cada usuario.
3. El usuario establece su contraseña al aceptar la invitación.

El sistema intenta autenticar vía Supabase Auth primero. Si falla (Auth no configurado, cuenta no existe), usa el `password` de `secrets.toml` como fallback automático.

### 4. Ejecutar en local

```bash
streamlit run app_principal.py
```

### 5. Desplegar en Streamlit Cloud

1. Conectar el repositorio en [share.streamlit.io](https://share.streamlit.io)
2. Entry point: `app_principal.py`
3. Añadir el contenido de `secrets.toml` en **App settings → Secrets**

---

## Roles y permisos

| Rol | Entrenamiento | Dashboard | Proyección Ingresos | Comparativa ML | Concesionarios | Escalabilidad |
|-----|:-------------:|:---------:|:-------------------:|:--------------:|:--------------:|:-------------:|
| `admin` | ✅ | ✅ (6 tabs) | ✅ | ✅ | ✅ | ✅ |
| `analyst` | ✅ | ✅ (6 tabs) | ✅ | ✅ | ✅ | ✅ |
| `financiero` | — | ✅ (2 tabs) | ✅ | — | — | ✅ |
| `manager` | — | ✅ (4 tabs) | — | — | ✅ | ✅ |
| `viewer` | — | ✅ (2 tabs) | — | — | — | ✅ |

**Tabs del Dashboard por rol:**

| Tab | Admin | Analista | Financiero | Gerente | Viewer |
|-----|:-----:|:--------:|:----------:|:-------:|:------:|
| 📊 Dashboard (KPIs + histórico) | ✅ | ✅ | ✅ | ✅ | ✅ |
| 🔮 Predicciones | ✅ | ✅ | ✅ | ✅ | ✅ |
| 💼 Recomendaciones de compra | ✅ | ✅ | — | ✅ | — |
| 🔄 Walk-Forward | ✅ | ✅ | — | — | — |
| 📋 Métricas Técnicas (sub-tabs: Resumen · 🔬 ACF/PACF · 🔍 Grid Search · 🏆 vs Descartados) | ✅ | ✅ | — | — | — |
| 🤖 Asistente IA (Gemini) | ✅ | ✅ | — | ✅ | — |

> El tab 🏪 Concesionarios se trasladó a la página independiente `pages/4_Concesionarios.py` (2026-05-28).
> Los tabs 🔬 ACF/PACF y 🔍 Grid Search se integraron como sub-pestañas de 📋 Métricas Técnicas (2026-05-28).
> Sub-tab 🏆 vs Descartados añadido a Métricas Técnicas con tabla metodológica de los 5 modelos (2026-05-30).

**Página independiente Proyección Ingresos (`pages/5_Proyeccion_Ingresos.py`):**

| Rol | Acceso | Exportar CSV |
|-----|:------:|:------------:|
| `admin` | ✅ | ✅ |
| `analyst` | ✅ | ✅ |
| `financiero` | ✅ | ✅ |
| `manager` | — | — |
| `viewer` | — | — |

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
8. [Entrenamiento]  Artefactos subidos a Supabase Storage + metadatos en PostgreSQL
9. [Entrenamiento]  Comparación con el modelo de producción actual
10. [Entrenamiento] Clic en "Aprobar" → activo=TRUE en DB + latest.txt actualizado + audit log
11. [Dashboard]     Carga el modelo activo automáticamente al arrancar
12. [Dashboard]     Watcher realtime: notifica via toast si se entrena un modelo nuevo
13. [Dashboard]     Barra lateral permite cambiar entre cualquier run histórico
14. [Proyección]    Ajusta precio, margen y tipo de cambio para ver el escenario financiero
15. [Comparativa]   Carga el mismo histórico + exog del run activo y enfrenta 5 modelos (SARIMAX idéntico a producción) en un solo clic
```

Sin ZIPs. Sin copias manuales. El entrenamiento escribe directamente en Supabase y el dashboard lee desde allí.

---

## App 1 — Entrenamiento (`pages/1_Entrenamiento.py`)

### Formato de datos de entrada

| Tipo | Hoja requerida | Se guarda en |
|------|---------------|--------------|
| Ventas | `Hoja1` | `session_state['df_raw']` |
| Stock | `Stock Actual` | `session_state['df_stock']` |

**Columnas esperadas en el Excel de ventas (`Hoja1`):**

| Columna | Descripción |
|---------|-------------|
| `FECHA-VENTA` | Fecha de venta (parseable por pandas) |
| `MARCA` | Marca del vehículo |
| `MODELO3` | Nombre del modelo |

**Columna adicional en `data/processed/veh_ml_features.xlsx`:**

| Columna | Descripción |
|---------|-------------|
| `ventas_otros` | Conteo mensual de ventas de **otros modelos de la misma marca** ese mes (`total_marca – modelo_propio`). Rango 0–202 uds/mes. Permite seleccionarla directamente como variable exógena al subir este archivo en la página Comparativa ML. |

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
| Fecha inicio | `2024-01-01` | Inicio de la ventana de entrenamiento |
| Horizonte | `6` meses | Meses a predecir (3–12) |
| Máx. ventas | `100` unid./mes | Límite superior de predicciones válidas |
| Excluir mes actual | `true` | Eliminar el mes en curso (datos incompletos) |

### Modelo SARIMA

- **Algoritmo**: SARIMAX con variable exógena opcional (ventas de otros modelos de la misma marca); se incluye automáticamente solo si `|Pearson r| ≥ 0.3` — si no, se entrena SARIMA puro para evitar inyección de ruido
- **Proyección exógena**: tendencia lineal (`polyfit` grado 1 sobre los últimos 12 meses) proyectada al horizonte; el sistema muestra pendiente, dirección (↗/↘/→) y rango
- **Búsqueda**: Optuna TPE (80 trials) sobre `p∈{0–3}`, `d∈{0–1}`, `q∈{0–3}`, `P∈{0–1}`, `D∈{0–1}`, `Q∈{0–2}`, `m=12`
- **Criterio**: MAPE mínimo sobre el conjunto de test
- **Walk-forward**: valida los últimos 12 meses (mínimo = horizonte configurado)
- **Intervalos de confianza**: 95% en todos los puntos del forecast
- **Trazabilidad exógena**: `pearson_r` y `usada: bool` guardados en `metricas_mejoradas.json` por cada run

---

## App 2 — Comparativa ML (`pages/2_Comparativa_ML.py`)

**Flujo de la página (6 secciones):**

| # | Sección | Descripción |
|---|---------|-------------|
| 1 | Fuente de datos | Carga desde un run de Supabase o sube un Excel manual |
| 2 | Período de análisis | Selectboxes Desde/Hasta para recortar el histórico antes de comparar |
| 3 | Configuración | Partición train/test, modelos a activar y parámetros SARIMAX |
| 4 | Ejecutar comparación | Botón único — entrena todos los modelos seleccionados en paralelo |
| 5 | Resultados | Tabla de métricas, gráficas, feature importances y tabla detallada |
| 6 | Publicar modelo ganador | Activa el run ganador en el Dashboard con un clic |

**Modelos comparados:**

| Modelo | Tipo | Enfoque |
|--------|------|---------|
| **SARIMAX** | Serie de tiempo | Idéntico al modelo de producción — misma exog `ventas_otros`, parámetros auto-cargados desde el run seleccionado |
| **Prophet** | Serie de tiempo | Estacionalidad multiplicativa anual + festivos PE |
| **Regresión Lineal** | ML supervisado | Lag features + rolling stats + calendario |
| **Random Forest** | ML supervisado | 300 estimadores |
| **XGBoost** | ML supervisado | Gradient boosting, lr=0.05, max_depth=4 |

**Métricas comparadas:** MAE, RMSE, MAPE (criterio principal), R², Tiempo (s).

> **Apples-to-apples desde v22 (2026-05-27):** la comparativa usa exactamente el mismo SARIMAX que Entrenamiento — misma variable exógena (`ventas_otros`), mismos parámetros Optuna y mismo split hold-out. Los parámetros (p,d,q)(P,D,Q) se auto-completan desde `metricas_mejoradas.json` del run cargado. Para runs anteriores a v22 sin `historico_exog.xlsx`, Comparativa muestra un aviso y ejecuta SARIMAX sin exog; se recomienda reentrenar para obtener la comparación completa.

---

## App 3 — Dashboard (`pages/3_Dashboard.py`)

### Selector de versión y Realtime

La barra lateral lista todos los runs disponibles (fuente: tabla `training_runs` en PostgreSQL). El run activo (`activo=TRUE`) aparece con 🟢. Un watcher con `@st.fragment(run_every=30)` detecta nuevos entrenamientos y muestra un `st.toast()` de notificación con recarga automática del selector.

### Tabs por rol

| Tab | Admin | Analista | Financiero | Gerente | Viewer |
|-----|:-----:|:--------:|:----------:|:-------:|:------:|
| 📊 Dashboard (KPIs + histórico) | ✅ | ✅ | ✅ | ✅ | ✅ |
| 🔮 Predicciones (N meses + IC + walk-forward overlay) | ✅ | ✅ | ✅ | ✅ | ✅ |
| 💼 Recomendaciones de compra (+ análisis ciclo estacional) | ✅ | ✅ | — | ✅ | — |
| 🔄 Walk-forward validation (detalle técnico) | ✅ | ✅ | — | — | — |
| 📋 Métricas Técnicas (sub-tabs: Resumen · 🔬 ACF/PACF · 🔍 Grid Search · 🏆 vs Descartados) | ✅ | ✅ | — | — | — |
| 🤖 **Asistente IA (Gemini)** | ✅ | ✅ | — | ✅ | — |

> La proyección financiera (💰 Proyección Ingresos) se trasladó a la página independiente `pages/5_Proyeccion_Ingresos.py` (2026-05-27).
> El tab 🏪 Concesionarios se trasladó a la página independiente `pages/4_Concesionarios.py` (2026-05-28).
> Los tabs 🔬 ACF/PACF y 🔍 Grid Search se consolidaron como sub-pestañas de 📋 Métricas Técnicas (2026-05-28).
> Sub-tab 🏆 vs Descartados y análisis de ciclo estacional añadidos (2026-05-30).

### Tab Recomendaciones de Compra — análisis de ciclo estacional

El tab incluye ahora un bloque de análisis estacional previo al marco teórico:

- **KPIs estacionales**: mes pico histórico, mes valle y ratio pico/valle calculados desde los datos reales del run activo.
- **Gráfico de media mensual**: barras con color coding (rojo = máximo, azul = sobre la media histórica, gris = bajo la media).
- **Callout de negocio**: explica el efecto rappel del proveedor en diciembre y la oportunidad operativa de des-estacionalizar los pedidos.

---

### Tab Predicciones — conceptos y secciones

#### Banner de contexto (dos conceptos diferenciados)

- **① Predicción mes a mes**: el modelo genera una estimación independiente para *cada mes* del horizonte. Cada fila de la tabla tiene su propio intervalo de confianza al 95% — no es un reparto del total.
- **② Horizonte de 6 meses**: ventana de visibilidad hacia adelante. En operación real, el equipo actualiza el histórico cada mes con las ventas cerradas y relanza la predicción. La **zona violeta** (`#A78BFA`) del gráfico muestra la validación walk-forward: el modelo predijo cada mes **un paso adelante** con todos los datos anteriores, simulando exactamente ese flujo. Es la estimación más honesta del MAPE real del sistema.

#### Gráfico principal

- **Zona violeta** (`#A78BFA`): período de validación walk-forward.
- **Línea futura** (rojo signal): predicción hacia los próximos N meses con banda IC 95%.
- **KPI "MAPE real (1 mes)"**: precisión del caso de uso real, objetivo < 15%.

---

## App 5 — Proyección de Ingresos (`pages/5_Proyeccion_Ingresos.py`)

Página independiente disponible para **Admin**, **Analista** y **Financiero** (permiso `ver_ingresos`). Traduce la predicción SARIMA del modelo activo en cifras financieras en dólares.

| Elemento | Descripción |
|----------|-------------|
| **Precio por unidad (USD $)** | Input configurable (default 15 000 $) |
| **Margen neto (%)** | Input opcional — si > 0 añade KPI y columna de beneficio |
| **Tipo de cambio** | Factor multiplicador para convertir a moneda local (default 1.0) |
| **KPIs** | Unidades totales, Ingresos centrales, Rango IC 95% en $, Beneficio estimado (si margen > 0) |
| **Gráfico de barras** | Barras de ingresos proyectados con banda IC 95% superpuesta (overlay) + línea de beneficio neto |
| **Tabla mes a mes** | Predicción (uds) · Ingresos · IC inferior/superior · Beneficio (si aplica) + fila de totales |
| **Exportar CSV** | Disponible para roles con permiso `exportar` |

Los ingresos se calculan multiplicando la predicción mensual por el precio efectivo (precio × tipo de cambio). El rango IC se traslada a dólares para comunicar la incertidumbre financiera. La visualización usa barras en overlay: la banda IC 95% aparece como capa semitransparente sobre las barras de ingreso central.

### Calculadora de ROI estratégico

Sección añadida al final de la página: **"Valor Estratégico del Sistema — ¿Cuánto vale predecir bien?"**

| Input | Por defecto | Descripción |
|-------|-------------|-------------|
| Sobrestock actual | 5 uds/mes | Unidades medias en exceso por mes (capital inmovilizado) |
| Costo de financiamiento (%) | 1.5% | Costo mensual del capital inmovilizado |
| Stockouts mensuales | 2 uds/mes | Ventas perdidas por rotura de stock |
| Reducción sobrestock (%) | 60% | % de mejora esperada con el sistema |
| Reducción stockout (%) | 70% | % de oportunidades recuperadas |
| Costo anual del sistema | 1 200 $ | Coste total de operar el sistema por año |

Outputs:
- **Waterfall chart**: Ahorro sobrestock → Ingresos recuperados → Valor bruto → Costo sistema → Valor neto.
- **4 KPIs**: ahorro por sobrestock anual, ingresos recuperados, valor neto anual, ROI multiplier (x veces el costo).
- **Tablas comparativas** "✅ Con sistema" vs "❌ Sin sistema".

> Extraída del tab 3 de `2_Dashboard.py` el 2026-05-27 para mejorar la navegación y separar la vista financiera de la operativa. Acceso restringido a roles con permiso `ver_ingresos`: admin, analyst y financiero (2026-05-27). Calculadora ROI añadida en 2026-05-30.

---

## App 4 — Concesionarios (`pages/4_Concesionarios.py`)

Página independiente disponible para **Admin**, **Analista** y **Manager**. Combina el análisis histórico de ventas desagregado por tienda con predicciones SARIMA distribuidas mediante shares.

**Metodología:** el modelo SARIMA predice el total nacional de ventas TIGGO 2. Para desglosar por concesionario se calcula el share de los últimos 12 meses de cada tienda y se aplica como ponderación sobre la predicción total y sus IC 95%.

| Tab | Descripción |
|-----|-------------|
| 📊 Resumen | Barras horizontales de ventas totales + mix de modelos apilado por concesionario |
| 📈 Evolución Mensual | Líneas por concesionario, share % 100% stacked area, variación MoM agrupada |
| 🔮 Predicciones por Tienda | KPIs próximo mes, gráfico histórico + predicción con IC 95%, barras de horizonte completo, tabla exportable |
| 📋 Detalle | Ranking acumulado + pivot mensual exportable |

**Editor de shares:** expander inline con `st.data_editor` para ajustar el % de cada concesionario y simular escenarios (apertura/cierre de tiendas, campañas locales). Los shares editados se renormalizan automáticamente y se aplican a todas las predicciones.

---

## App 6 — Escalabilidad (`pages/6_Escalabilidad.py`)

Página disponible para **todos los roles**. Presenta la hoja de ruta para exportar el pipeline a otras marcas, líneas de negocio y mercados.

| Tab | Descripción |
|-----|-------------|
| 🏗️ Arquitectura | Diagrama del stack técnico y flujo de datos |
| 🚗 Portafolio | Hoja de ruta de expansión a otros modelos del portafolio Chery |
| 💼 Líneas de Negocio | Aplicación del sistema a flotillas, leasing y postventa |
| 📋 Playbook de Onboarding | Pasos para incorporar un nuevo modelo o marca en < 2 semanas |
| 🌎 Expansión Geográfica | Roadmap de despliegue en otros mercados LatAm |
| 🚀 Visión del Producto | Evolución de 3 etapas: Reactivo → Proactivo → Autónomo |

### Tab Visión del Producto — evolución del sistema

| Etapa | Período | Estado | Capacidades |
|-------|---------|--------|-------------|
| **Etapa 1 — Reactivo** | HOY 2025-2026 | EN PRODUCCIÓN | Predicción SARIMA mensual, dashboard, RBAC, Gemini |
| **Etapa 2 — Proactivo** | AÑO 1 2026-2027 | ROADMAP | Auto-retraining, multi-brand, integración ERP, alertas push |
| **Etapa 3 — Autónomo** | AÑO 2+ 2027-2028 | VISIÓN | SaaS multi-tenant, optimización de precio, API LatAm |

Gráfico dual-axis: barras de valor de negocio + línea de autonomía operativa por etapa.

---

## Módulo `core/`

| Módulo | Responsabilidad |
|--------|----------------|
| `core/auth_system.py` | Autenticación Supabase Auth + fallback SHA-256, sesiones, timeout, RBAC, UI de login |
| `core/supabase_io.py` | I/O con Supabase Storage y PostgreSQL; audit log centralizado; caché `@st.cache_data` en todas las funciones de lectura (TTL 5–10 min), invalidación automática al entrenar o aprobar un modelo |
| `core/utils_validacion.py` | Validación de calidad del DataFrame antes de entrenar |
| `core/logger.py` | Logger centralizado — consola + `logs/app.log` (rotativo 2 MB × 3) |
| `core/styles.py` | CSS global dark premium, helpers `kpi_card()`, `section_header()`, `apply_chart_theme()` |

---

## Seguridad

- Autenticación vía **Supabase Auth** (email + password gestionado por Supabase); fallback a hash SHA-256 en `secrets.toml`
- Todas las acciones críticas (login, logout, aprobación, borrado) quedan registradas en la tabla `audit_log`
- Timeout de sesión configurable (60 min por defecto)
- Credenciales nunca en el código fuente — solo en `st.secrets`
- Bucket de Supabase **privado** — acceso exclusivamente vía anon key del SDK

---

## Tests

```bash
pytest tests/ -v
```

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

## Modelo activo (2026-05-30)

| Métrica | Valor |
|---------|-------|
| **Run ID** | `20260530_103626` |
| **Período de datos** | 2022-01 → 2026-03 |
| **Meses de datos** | 51 |
| **Total ventas históricas** | 1 804 uds |
| **Fecha inicio entrenamiento** | 2022-01-01 |
| **Horizonte de pronóstico** | 6 meses |
| **Orden SARIMA** | `(2, 0, 1)(1, 0, 2)[12]` |
| **AIC** | 138.48 |
| **BIC** | 190.85 |
| **Variable exógena (`ventas_otros`)** | ✅ usada (Pearson r = 0.581) |
| **MAPE walk-forward** | **14.65%** |
| **Meses validados (walk-forward)** | 12 |
| **Trials válidos / total** | 62 / 80 |
| **Combinaciones descartadas** | 18 |

---

## Licencia

Uso interno. No distribuir públicamente.
