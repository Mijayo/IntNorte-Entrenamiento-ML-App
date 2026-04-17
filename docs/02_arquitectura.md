# Arquitectura del Sistema

## Visión general

El sistema sigue una arquitectura **modular de tres capas**: interfaz (Streamlit), lógica de negocio (Python) y almacenamiento persistente (Supabase). El almacenamiento tiene dos componentes complementarios: un **bucket de objetos** (Storage) para los artefactos de cada modelo y una **base de datos relacional** (PostgreSQL) para los metadatos de runs y el audit log.

```
┌──────────────────────────────────────────────────────────────┐
│                      STREAMLIT CLOUD                         │
│                                                              │
│  app_principal.py  ←── Entry point / autenticación          │
│       │                                                      │
│       ├── pages/1_Entrenamiento.py                           │
│       ├── pages/2_Dashboard.py   ← realtime watcher         │
│       └── pages/3_Comparativa_ML.py                         │
│                                                              │
│  Módulos compartidos (core/):                                │
│  ├── auth_system.py       (Supabase Auth + sesiones + RBAC) │
│  ├── supabase_io.py       (Storage + PostgreSQL + Audit Log) │
│  ├── utils_validacion.py  (validación de datos)              │
│  ├── logger.py            (logging centralizado)             │
│  └── styles.py            (CSS global)                       │
└───────────────────┬──────────────────────────────────────────┘
                    │ HTTPS / supabase-py SDK
┌───────────────────▼──────────────────────────────────────────┐
│                        SUPABASE                              │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  STORAGE (bucket: modelos-ml)                       │    │
│  │  latest.txt               ← run activo (backup)     │    │
│  │  training_log.json        ← historial (backup)      │    │
│  │  YYYYMMDD_HHMMSS/         ← carpeta por run         │    │
│  │    ├── metricas_mejoradas.json                      │    │
│  │    ├── prediccion_total_mejorada.xlsx               │    │
│  │    ├── grid_search_results.xlsx                     │    │
│  │    ├── walk_forward_validation.xlsx                 │    │
│  │    ├── historico_total_mejorado.xlsx                │    │
│  │    ├── modelo_total_mejorado.pkl.gz                 │    │
│  │    ├── acf_plot.png / pacf_plot.png                 │    │
│  │    └── llm_cache.json   ← caché Gemini por run      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  POSTGRESQL                                         │    │
│  │  training_runs   ← registro primario de runs        │    │
│  │  audit_log       ← trazabilidad de acciones         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  AUTH                                               │    │
│  │  users (email + password gestionados por Supabase)  │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

> No se persiste ningún dato en disco local. Todo el almacenamiento es en la nube, lo que permite que la aplicación sea **stateless** y se pueda escalar o reiniciar sin pérdida de información.

---

## Módulos del sistema

### `app_principal.py` — Entry point

Página de inicio de la aplicación Streamlit. Responsabilidades:
1. Inicializar el session state.
2. Verificar si el usuario está autenticado (delegando en `auth_system`).
3. Si no, mostrar el formulario de login.
4. Si sí, presentar la página de bienvenida con accesos directos según el rol.

---

### `auth_system.py` — Autenticación y permisos

Gestiona el ciclo de vida de la sesión. Implementa un flujo de autenticación en dos etapas:

1. **Supabase Auth (primario):** si el usuario tiene configurado un `email` en `secrets.toml`, el sistema llama a `supabase.auth.sign_in_with_password()`. El access token resultante se almacena en `session_state` para invalidación en logout.
2. **Fallback local (secundario):** si Supabase Auth no está configurado o no está disponible, el sistema verifica las credenciales contra el campo `password` de `secrets.toml` (texto plano o hash SHA-256).

Los roles y permisos se leen siempre de `secrets.toml`, no de Supabase Auth, para mantener un único punto de configuración de RBAC.

**Funciones principales:**

| Función | Descripción |
|---------|-------------|
| `init_session_state()` | Inicializa `authenticated`, `username`, `role`, `permissions` |
| `login(username, password)` | Intenta Supabase Auth; fallback a credenciales locales |
| `logout()` | Llama a `supabase.auth.sign_out()`, registra en audit log, limpia session_state |
| `check_session_timeout()` | Invalida la sesión si supera 30 minutos de inactividad |
| `has_permission(name)` | Retorna `True/False` para un permiso específico |
| `show_header(title, subtitle)` | Renderiza el encabezado corporativo |
| `show_login_page(title)` | Renderiza el formulario de login con logo corporativo |
| `show_user_info()` | Card de usuario en sidebar con badge de rol y countdown de sesión |

**Ejemplo de verificación de permiso:**
```python
if not has_permission('entrenar_modelos'):
    st.error("❌ No tienes permiso para acceder a esta sección")
    st.stop()
```

**Permisos disponibles:**

| Permiso | Descripción |
|---------|-------------|
| `entrenar_modelos` | Acceso a Entrenamiento y Comparativa ML |
| `ver_dashboard` | Acceso al Dashboard |
| `exportar` | Descarga de predicciones en CSV |

---

### `supabase_io.py` — Capa de I/O centralizada

Toda la comunicación con Supabase pasa por este módulo. Ninguna página importa el cliente de Supabase directamente. El módulo gestiona tres capas de Supabase:

- **Storage** — subida y descarga de artefactos binarios (modelos, Excel, imágenes, JSON)
- **PostgreSQL** — lectura y escritura en tablas `training_runs` y `audit_log`
- **Auth** — el cliente compartido se usa también por `auth_system.py`

**Estrategia dual-write:** la tabla `training_runs` es la fuente primaria para gestión de runs; `training_log.json` en Storage se mantiene como backup de fallback.

**Funciones principales:**

| Función | Descripción |
|---------|-------------|
| `get_client()` | Cliente Supabase cacheado (`@st.cache_resource`) |
| `save_to_dashboard(run_name, ...)` | Sube todos los artefactos del run a Storage |
| `load_precargados(run_name)` | Descarga y parsea artefactos de un run (cacheado 10 min) |
| `get_available_runs()` | Lista runs desde PostgreSQL (fallback: `training_log.json`), filtrando los que tienen artefactos en Storage |
| `get_default_run(runs)` | Run activo: primero `activo=TRUE` en DB, luego `latest.txt` |
| `approve_model(run_name, usuario)` | Marca `activo=TRUE` en DB + actualiza `latest.txt` + registra en audit log |
| `delete_run(run_name, usuario)` | Elimina de DB + registra en audit log |
| `save_training_log(entry)` | Upsert en `training_runs` + backup en `training_log.json` |
| `load_training_log()` | Historial completo desde DB (fallback: JSON) |
| `get_runs_df()` | DataFrame con todos los runs y métricas para análisis comparativo |
| `log_audit(usuario, accion, run_name, detalle)` | Registra acción en `audit_log`; falla silenciosamente |
| `get_audit_log(limit)` | Últimas N entradas del audit log |
| `save_llm_cache(run_name, cache)` | Persiste caché de respuestas Gemini en Storage |
| `load_llm_cache(run_name)` | Descarga caché Gemini; devuelve `{}` si no existe |
| `load_acf_pacf_images(run_name)` | Imágenes ACF/PACF como bytes para `st.image` |

**Esquema de tablas PostgreSQL:**

```sql
-- Fuente primaria de todos los runs de entrenamiento
CREATE TABLE training_runs (
  id                        BIGSERIAL    PRIMARY KEY,
  run_name                  TEXT         NOT NULL UNIQUE,  -- 'YYYYMMDD_HHMMSS'
  timestamp                 TIMESTAMPTZ,
  usuario                   TEXT,
  modelo                    TEXT,
  marca                     TEXT,
  fecha_inicio              TEXT,
  horizonte                 INT,
  max_ventas                INT,
  sarima_order              TEXT,         -- JSON serializado: "[1, 1, 1]"
  sarima_seasonal           TEXT,         -- JSON serializado: "[1, 1, 1, 12]"
  aic                       NUMERIC(10,2),
  mape_wf                   NUMERIC(6,2),
  meses_datos               INT,
  combinaciones_validas     INT,
  combinaciones_descartadas INT,
  activo                    BOOLEAN      NOT NULL DEFAULT FALSE,
  created_at                TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
-- Garantiza que solo un run puede ser activo a la vez
CREATE UNIQUE INDEX idx_training_runs_activo ON training_runs (activo) WHERE activo = TRUE;

-- Trazabilidad de acciones de usuario
CREATE TABLE audit_log (
  id        BIGSERIAL    PRIMARY KEY,
  timestamp TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  usuario   TEXT,
  accion    TEXT         NOT NULL,  -- 'LOGIN', 'LOGOUT', 'APPROVE_MODEL', 'DELETE_RUN'
  run_name  TEXT,                   -- nullable; relacionado con training_runs.run_name
  detalle   JSONB                   -- información adicional (metodo auth, etc.)
);
```

**Ejemplo de carga de un modelo:**
```python
import core.supabase_io as sio

runs = sio.get_available_runs()          # ['20260417_143000', '20260410_091500']
run  = sio.get_default_run(runs)         # '20260417_143000'
metricas, pred, grid, wf, hist = sio.load_precargados(run)

print(metricas['mejor_modelo']['order'])              # [1, 1, 1]
print(metricas['walk_forward_validation']['mape'])    # 5.34
print(pred.head())
#         Fecha       Mes  Predicción  IC_Inferior  IC_Superior
# 0  2026-05-31  mayo 2026        64.2         52.1         76.3
```

---

### `utils_validacion.py` — Validación de datos

Valida la calidad del Excel de ventas antes de lanzar el entrenamiento.

| Check | Umbral | Acción si falla |
|-------|--------|-----------------|
| Columnas requeridas (`FECHA-VENTA`, `MODELO3`, `MARCA`) | — | Error bloqueante |
| Fechas parseables | Máx. 5% inválidas | Error bloqueante |
| Período mínimo | ≥ 36 meses | Error bloqueante |
| Datos faltantes por columna | Máx. 5% | Advertencia |

---

## Flujo de datos: Entrenamiento → Dashboard

```
Excel de ventas
      │
      ▼
[Tab 1: Cargar]
  ├── Elimina duplicados por CHASIS (conserva el más reciente)
  └── Elimina filas con MODELO3 nulo
      │
      ▼
[Tab 2: Validación]
  └── Verifica columnas, fechas, período mínimo, nulos
      │
      ▼
[Tab 4: Entrenamiento]
  ├── Filtra por marca + modelo + rango de fechas
  ├── Resample mensual → Serie temporal
  ├── Construye variable exógena (ventas otros modelos)
  ├── Test ADF (estacionariedad)
  ├── Optuna TPE (80 trials) → mejores parámetros SARIMA
  ├── Walk-forward validation (últimos 12 meses)
  ├── Modelo final + forecast N meses
  └── save_to_dashboard()  →  Storage (artefactos)
      save_training_log()  →  PostgreSQL (metadatos)
      log_audit("ENTRENAMIENTO")
      │
      ▼
[Tab 5: Comparación]
  └── Compara MAPE/AIC nuevo vs activo en DB
      │ si "Aprobar"
      ▼
  approve_model(run_name, usuario)
  ├── UPDATE training_runs SET activo=TRUE WHERE run_name=X
  ├── UPDATE training_runs SET activo=FALSE WHERE run_name!=X
  ├── _upload("latest.txt", run_name)
  └── log_audit("APPROVE_MODEL")
      │
      ▼
[Dashboard — @st.fragment(run_every=30)]
  └── get_available_runs() → detecta run nuevo → st.toast() + st.rerun()
      │
      ▼
[Dashboard — página principal]
  └── load_precargados(run_activo) → muestra predicciones y KPIs
```

---

## Gestión de versiones de modelos

Cada ejecución de entrenamiento genera un **run** identificado por timestamp (`YYYYMMDD_HHMMSS`). Los runs se registran en la tabla `training_runs` y sus artefactos en Storage.

El campo `activo=TRUE` (con índice único parcial) garantiza que solo **un run puede ser el modelo de producción** en cualquier momento. El fichero `latest.txt` se mantiene como redundancia para el caso de que la DB no esté disponible.

Esto permite:
- Revertir a una versión anterior activando el run anterior en DB
- Consultar cualquier run histórico desde el Dashboard sin afectar producción
- Auditar quién entrenó y aprobó cada modelo vía `audit_log`

---

## Notificaciones Realtime

El Dashboard incluye un fragmento autónomo que se reejcuta automáticamente cada 30 segundos:

```python
@st.fragment(run_every=30)
def _live_watcher():
    current = sio.get_available_runs()
    prev    = st.session_state.get('_known_runs', [])
    new     = [r for r in current if r not in prev]
    if new:
        for r in new:
            st.toast(f"Nuevo modelo: {sio.format_run_label(r)}", icon="🔔")
        st.session_state['_known_runs'] = current
        st.rerun()
```

Al detectar un run nuevo, muestra un `st.toast()` y fuerza un `st.rerun()` para actualizar el selector de versión sin que el usuario tenga que recargar la página.

---

## Consideraciones de seguridad

- Las contraseñas de usuarios se gestionan en **Supabase Auth** (hash bcrypt, nunca expuesto). El fallback local usa hash SHA-256 en `secrets.toml`.
- Todas las acciones críticas quedan registradas en `audit_log` con usuario, timestamp y detalle JSON.
- El fichero `.streamlit/secrets.toml` está en `.gitignore` y nunca se sube al repositorio.
- El bucket de Supabase debe ser **privado** — el acceso se realiza exclusivamente con la anon key del SDK.
- El timeout de sesión (30 min) limita la exposición de sesiones inactivas.
- El sistema no expone datos de ventas individualmente — solo series temporales agregadas por mes.
