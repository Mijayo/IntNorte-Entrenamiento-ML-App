# Sistema TIGGO 2 — Predicción de Ventas con ML

Sistema multipage en Streamlit para entrenar modelos SARIMA, comparar múltiples algoritmos de predicción (SARIMA, Prophet, Regresión Lineal, Random Forest, XGBoost) y visualizar predicciones de demanda del Chery Tiggo 2. Desplegado en **Streamlit Cloud** con almacenamiento persistente en **Supabase Storage**, base de datos relacional en **Supabase PostgreSQL** y autenticación via **Supabase Auth**.

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
├── auth_system.py            ← Autenticación Supabase Auth + fallback local, sesiones, RBAC
├── supabase_io.py            ← Capa de I/O centralizada (Storage + PostgreSQL + Audit Log)
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
entrenar_modelos = true
ver_dashboard    = true
exportar         = true
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

| Rol | Entrenamiento | Comparativa ML | Tabs del Dashboard |
|-----|:-------------:|:--------------:|---------------------|
| `admin` | ✅ | ✅ | Dashboard, Predicciones, **Proyección Ingresos**, ACF/PACF, Grid Search, Walk-Forward, Métricas técnicas, **Asistente IA**, Concesionarios |
| `analyst` | ✅ | ✅ | Dashboard, Predicciones, **Proyección Ingresos**, ACF/PACF, Grid Search, Walk-Forward, Métricas técnicas, **Asistente IA**, Concesionarios |
| `manager` | — | — | Dashboard, Predicciones, **Proyección Ingresos**, Recomendaciones de compra, **Asistente IA**, Concesionarios |
| `viewer` | — | — | Dashboard, Predicciones, **Proyección Ingresos** |

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
14. [Comparativa]   Carga el mismo histórico y enfrenta 5 modelos en un solo clic
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

## App 3 — Comparativa ML (`pages/3_Comparativa_ML.py`)

| Modelo | Tipo | Enfoque |
|--------|------|---------|
| **SARIMA** | Serie de tiempo | Parámetros (p,d,q)(P,D,Q)₁₂ configurables |
| **Prophet** | Serie de tiempo | Estacionalidad multiplicativa anual + festivos PE |
| **Regresión Lineal** | ML supervisado | Lag features + rolling stats + calendario |
| **Random Forest** | ML supervisado | 300 estimadores |
| **XGBoost** | ML supervisado | Gradient boosting, lr=0.05, max_depth=4 |

**Métricas comparadas:** MAE, RMSE, MAPE (criterio principal), R², Tiempo (s).

---

## App 2 — Dashboard (`pages/2_Dashboard.py`)

### Selector de versión y Realtime

La barra lateral lista todos los runs disponibles (fuente: tabla `training_runs` en PostgreSQL). El run activo (`activo=TRUE`) aparece con 🟢. Un watcher con `@st.fragment(run_every=30)` detecta nuevos entrenamientos y muestra un `st.toast()` de notificación con recarga automática del selector.

### Tabs por rol

| Tab | Admin | Analista | Gerente | Viewer |
|-----|:-----:|:--------:|:-------:|:------:|
| Dashboard (KPIs + histórico) | ✅ | ✅ | ✅ | ✅ |
| Predicciones (N meses + IC + walk-forward overlay) | ✅ | ✅ | ✅ | ✅ |
| **Proyección Ingresos (USD, horizonte 6 meses)** | ✅ | ✅ | ✅ | ✅ |
| Recomendaciones de compra | — | — | ✅ | — |
| Análisis ACF/PACF | ✅ | ✅ | — | — |
| Resultados Grid Search | ✅ | ✅ | — | — |
| Walk-forward validation (detalle técnico) | ✅ | ✅ | — | — |
| Métricas técnicas completas | ✅ | ✅ | — | — |
| **Asistente IA (Gemini)** | ✅ | ✅ | ✅ | — |
| Ventas por Concesionario | ✅ | ✅ | ✅ | — |

### Tab Predicciones — conceptos y secciones

#### Banner de contexto (dos conceptos diferenciados)

- **① Predicción mes a mes**: el modelo genera una estimación independiente para *cada mes* del horizonte. Cada fila de la tabla tiene su propio intervalo de confianza al 95% — no es un reparto del total.
- **② Horizonte de 6 meses**: ventana de visibilidad hacia adelante. En operación real, el equipo actualiza el histórico cada mes con las ventas cerradas y relanza la predicción. La **zona violeta** (`#A78BFA`) del gráfico muestra la validación walk-forward: el modelo predijo cada mes **un paso adelante** con todos los datos anteriores, simulando exactamente ese flujo. Es la estimación más honesta del MAPE real del sistema.

#### Gráfico principal

- **Zona violeta** (`#A78BFA`): período de validación walk-forward.
- **Línea futura** (rojo signal): predicción hacia los próximos N meses con banda IC 95%.
- **KPI "MAPE real (1 mes)"**: precisión del caso de uso real, objetivo < 15%.

### Tab Proyección de Ingresos — USD, Horizonte 6 Meses

Tab dedicado disponible para **todos los roles**. Traduce la predicción SARIMA en cifras financieras en dólares.

| Elemento | Descripción |
|----------|-------------|
| **Precio por unidad (USD $)** | Input configurable (default 27 000 $) |
| **Margen neto (%)** | Input opcional — si > 0 añade KPI y columna de beneficio |
| **Tipo de cambio** | Factor multiplicador para convertir a moneda local (default 1.0) |
| **KPIs** | Unidades totales, Ingresos centrales, Rango IC 95% en $, Beneficio estimado (si margen > 0) |
| **Gráfico de barras** | Barras de ingresos proyectados con banda IC 95% superpuesta (overlay) + línea de beneficio neto |
| **Tabla mes a mes** | Predicción (uds) · Ingresos · IC inferior/superior · Beneficio (si aplica) + fila de totales |
| **Exportar CSV** | Disponible para roles con permiso `exportar` |

Los ingresos se calculan multiplicando la predicción mensual por el precio efectivo (precio × tipo de cambio). El rango IC se traslada a dólares para comunicar la incertidumbre financiera. La visualización usa barras en overlay: la banda IC 95% aparece como capa semitransparente sobre las barras de ingreso central.

---

## Módulo `core/`

| Módulo | Responsabilidad |
|--------|----------------|
| `core/auth_system.py` | Autenticación Supabase Auth + fallback SHA-256, sesiones, timeout, RBAC, UI de login |
| `core/supabase_io.py` | I/O con Supabase Storage y PostgreSQL; audit log centralizado |
| `core/utils_validacion.py` | Validación de calidad del DataFrame antes de entrenar |
| `core/logger.py` | Logger centralizado — consola + `logs/app.log` (rotativo 2 MB × 3) |
| `core/styles.py` | CSS global dark premium, helpers `kpi_card()`, `section_header()`, `apply_chart_theme()` |

---

## Seguridad

- Autenticación vía **Supabase Auth** (email + password gestionado por Supabase); fallback a hash SHA-256 en `secrets.toml`
- Todas las acciones críticas (login, logout, aprobación, borrado) quedan registradas en la tabla `audit_log`
- Timeout de sesión configurable (30 min por defecto)
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

## Modelo activo (Iteración 3 — 2026-05-22)

| Métrica | Valor |
|---------|-------|
| **Orden SARIMA** | `(1, 1, 0)(1, 0, 2)[12]` |
| **AIC** | 137.38 |
| **MAPE walk-forward** | **10.32%** |
| **Trials válidos** | 71 / 80 |
| **Horizonte de pronóstico** | 6 meses |

---

## Licencia

Uso interno. No distribuir públicamente.
