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

#### Proyección de Ingresos · Horizonte 6 Meses

Sección separada debajo de las tablas de predicción:

| Elemento | Descripción |
|----------|-------------|
| **Precio por unidad (€)** | Input configurable (default 25 000 €) |
| **Margen neto (%)** | Input opcional — si > 0 añade columna de beneficio |
| **KPIs** | Unidades totales, Ingresos totales, Rango IC 95% en € |
| **Tabla mes a mes** | Predicción (uds) · Ingresos · IC inferior/superior · Beneficio (si margen > 0) |

Los ingresos se calculan multiplicando la predicción mensual por el precio unitario; el beneficio aplica el margen sobre los ingresos. El rango IC se traslada a euros para comunicar la incertidumbre en términos financieros.

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

## Changelog

### 2026-05-04 (v18)
- **feat**: **Clarificación conceptual en Tab Predicciones** — el banner storytelling ahora distingue explícitamente ① predicción mes a mes (cada fila es independiente con su propio IC 95%) vs ② horizonte de 6 meses (ventana de visibilidad + ciclo operativo de renovación mensual).
- **feat**: **Proyección de Ingresos · Horizonte 6 Meses** — nueva sección debajo de las tablas de predicción con inputs configurables de precio por unidad (€) y margen neto (%). Genera KPIs de ingresos totales y rango IC 95% en euros, tabla mes a mes con ingresos/beneficio y columna de beneficio opcional cuando el margen es > 0.

### 2026-04-30 (v17)
- **feat**: **Filtro de correlación del exógeno** — antes de entrenar calcula `Pearson r` entre `ventas_modelo` y `ventas_otros`. Si `|r| < 0.3` la variable exógena se descarta automáticamente y SARIMA entrena sin ella, eliminando ruido. El valor `pearson_r` y `usada: bool` quedan en `metricas_mejoradas.json`.
- **feat**: **Proyección exógena por tendencia lineal** — reemplaza la media móvil constante. `polyfit` grado 1 sobre los últimos 12 meses proyectado al horizonte; muestra pendiente, dirección (↗/↘/→) y rango proyectado.
- **fix**: `perform_walk_forward` y slicing de `train_exog`/`test_exog` manejan `exog_data=None` cuando el filtro de correlación descarta la variable.

### 2026-04-29 (v16)
- **feat**: **Walk-forward en tab Predicciones** — overlay violeta (`#A78BFA`) en el gráfico principal mostrando las predicciones 1-mes-adelante de la validación walk-forward, visible a todos los roles.
- **feat**: **Storytelling operacional** — banner explicativo en el tab Predicciones: "SARIMA puede proyectar 6 meses, pero el caso de uso real es predecir 1 mes y renovar mes a mes".
- **feat**: **KPI "MAPE real (1 mes)"** — nueva tarjeta en tab Predicciones con el MAPE walk-forward y semáforo de color.
- **feat**: **Tabla walk-forward en predicciones** — tabla con gradiente de error al lado de la tabla de predicciones futuras, para todos los roles.
- **feat**: Región sombreada + anotación "Validación walk-forward" sobre el período validado en el gráfico.
- **feat**: **Objetivo MAPE < 15%** — umbrales actualizados de 20% → 15% en Dashboard (Tab 0), Tab Predicciones y página de Entrenamiento.

### 2026-04-17 (v15)
- **feat**: **Supabase Auth** — `auth_system.py` autentica via `supabase.auth.sign_in_with_password()` usando el `email` del usuario configurado en `secrets.toml`. Fallback automático a credenciales locales SHA-256 si Supabase Auth no está disponible. Logout cierra la sesión en Supabase con `auth.sign_out()`.
- **feat**: **Audit Log** — nueva tabla `audit_log` en PostgreSQL. `log_audit()` registra `LOGIN`, `LOGOUT`, `APPROVE_MODEL` y `DELETE_RUN` con usuario, run y detalle JSON. `get_audit_log()` disponible para consultas.
- **feat**: **Realtime** — `@st.fragment(run_every=30)` en Dashboard detecta nuevos runs en Supabase cada 30 s y muestra `st.toast()` + recarga automática del selector.
- **feat**: **PostgreSQL como fuente primaria** — `get_available_runs()` y `get_default_run()` consultan la tabla `training_runs` (con fallback a `training_log.json`). `approve_model()` actualiza el campo `activo` en DB además de `latest.txt`. `save_training_log()` hace upsert en DB + backup JSON. Nuevas funciones `get_runs_df()` y `delete_run()`.
- **refactor**: `approve_model(run_name, usuario)` y `delete_run(run_name, usuario)` aceptan `usuario` para el audit log.

### 2026-04-16 (v14)
- **refactor**: Profesionalización de la estructura de carpetas — módulos Python extraídos a paquete `core/`.
- **refactor**: Importaciones actualizadas en todos los archivos.
- **fix**: `core/logger.py` — corregido `_LOGS_DIR` para escribir en la raíz del proyecto.
- **chore**: Datos locales reorganizados en `data/{raw,processed,monthly,artifacts}/`.

### 2026-04-16 (v13)
- **feat**: Constante `TRAINING_DEFAULT_START = date(2024, 1, 1)` — ventana de entrenamiento por defecto cambiada a 2024-01-01.
- **feat**: Nuevo expander "¿Cómo elegir la ventana de entrenamiento?" con tabla de casos de uso y alertas contextuales.
- **feat**: Tabla de diagnóstico de MAPE con 5 causas y soluciones.

### 2026-04-15 (v12)
- **refactor**: Magic numbers extraídos a constantes nombradas.
- **fix**: `warnings.filterwarnings` acotado solo a módulos statsmodels.
- **feat**: Validación anticipada de `max_ventas` antes de lanzar Optuna.
- **feat**: Límite de 500 caracteres en inputs del Asistente IA.

### 2026-04-15 (v11)
- **feat**: Nuevo módulo `logger.py` — logging centralizado.
- **feat**: Nueva suite de tests `tests/test_validacion.py` — 17 tests.
- **refactor**: Type hints completos en módulos `core/`.

### 2026-04-04 (v10)
- **fix**: Restricción `d=1 AND D=1` en Optuna.
- **feat**: Walk-forward extendido a 12 meses.
- **feat**: Alertas dinámicas de MAPE en Dashboard (rojo / ámbar / verde).
- **feat**: Caché Gemini persistido en Supabase (`<run>/llm_cache.json`).

### 2026-03-28 (v9)
- **feat**: Dark premium UI — `#080D18` bg, `#20C997` teal, `#F59E0B` amber.
- **feat**: Módulo `styles.py` con CSS global centralizado.

### 2026-03-27 (v8)
- **feat**: Nueva página **🏆 Comparativa ML** — SARIMA, Prophet, Regresión Lineal, Random Forest, XGBoost.

### 2026-03-25 (v7)
- **feat**: Búsqueda de hiperparámetros migrada a **Optuna TPE** (80 trials vs 384 combinaciones, ~4× más rápido).

### 2026-03-23 (v4–v6)
- **feat**: Asistente IA Gemini, tab Concesionarios, Comparativa Prophet vs SARIMA.

### 2026-03-23 (v1–v3)
- **feat**: MVP inicial — entrenamiento SARIMA, dashboard básico, autenticación por roles.

---

## Licencia

Uso interno. No distribuir públicamente.
