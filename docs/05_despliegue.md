# Guía de Despliegue

Esta guía explica cómo configurar el sistema en local para desarrollo y cómo desplegarlo en producción con **Streamlit Cloud**. No se incluyen credenciales reales — consulta al administrador del sistema para obtenerlas.

---

## Requisitos previos

- Python 3.10 o superior
- Cuenta en [Supabase](https://supabase.com) con un proyecto creado
- Cuenta en [Streamlit Cloud](https://share.streamlit.io) (o acceso al workspace del equipo)
- Clave API de [Google Gemini](https://ai.google.dev) (para el Asistente IA del Dashboard)
- Git

---

## Instalación en local

### 1. Clonar el repositorio

```bash
git clone https://github.com/<org>/IntNorte-Entrenamiento-ML-App.git
cd IntNorte-Entrenamiento-ML-App
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate       # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Las dependencias principales son:

```
streamlit       — Interfaz web
pandas          — Manipulación de datos
numpy           — Cálculo numérico
statsmodels     — Modelo SARIMAX
prophet         — Modelo Prophet (Meta)
optuna          — Optimización bayesiana de hiperparámetros
scikit-learn    — Métricas de error (MAE, RMSE)
plotly          — Gráficos interactivos
matplotlib      — Gráficos ACF/PACF
supabase        — Cliente Supabase (Storage + PostgreSQL + Auth)
google-genai    — Cliente Google Gemini (Asistente IA)
openpyxl        — Lectura/escritura de archivos Excel
pillow          — Procesamiento de imágenes (ACF/PACF)
xgboost         — Gradient boosting
```

> **Nota sobre Prophet:** en algunos sistemas es necesario instalar `pystan` o `cmdstanpy` antes de Prophet. Consulta la [documentación oficial](https://facebook.github.io/prophet/docs/installation.html) si la instalación falla.

---

### 3. Configurar Supabase

#### 3.1 Storage — Crear el bucket

1. En tu proyecto Supabase → **Storage → Buckets → New bucket**
2. Nombre: `modelos-ml` (debe coincidir con `bucket` en `secrets.toml`)
3. Visibilidad: **Private**

#### 3.2 PostgreSQL — Crear las tablas

En **Supabase → SQL Editor**, ejecuta:

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

-- Garantiza que solo un run puede ser activo a la vez
CREATE UNIQUE INDEX IF NOT EXISTS idx_training_runs_activo
  ON training_runs (activo)
  WHERE activo = TRUE;

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

#### 3.3 Auth — Crear usuarios

1. En Supabase → **Authentication → Users**
2. Haz clic en **Invite user** (o **Add user**)
3. Introduce el email de cada usuario que aparece en `secrets.toml`
4. El usuario recibirá un email de invitación y establecerá su contraseña

> El sistema intentará autenticar vía Supabase Auth primero. Si no está configurado o falla, usa automáticamente el campo `password` de `secrets.toml` como fallback.

---

### 4. Configurar credenciales

Copia la plantilla:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edita `.streamlit/secrets.toml` con los valores reales:

```toml
# API de Google Gemini (Asistente IA)
GENAI_API_KEY = "tu_clave_gemini_aqui"

[supabase]
url    = "https://<tu-proyecto>.supabase.co"
key    = "tu_anon_key_de_supabase"
bucket = "modelos-ml"

# ── Usuarios ──────────────────────────────────────────────────────────────────
# El campo 'email' activa Supabase Auth (recomendado).
# El campo 'password' se usa como fallback si Auth no está disponible.
# Ambos pueden coexistir durante la migración.

[users.admin]
email    = "admin@tudominio.com"      # cuenta en Supabase Auth
password = "hash_sha256_opcional"    # fallback local
role     = "admin"
name     = "Nombre Completo"
icon     = "👑"

[users.admin.permissions]
entrenar_modelos      = true
ver_metricas_tecnicas = true
ver_predicciones      = true
exportar              = true
gestionar_usuarios    = true
ver_grid_search       = true
ver_acf_pacf          = true
ver_ingresos          = true   # acceso a Proyección de Ingresos

[users.analista1]
email    = "analista@tudominio.com"
role     = "analyst"
name     = "Analista Uno"
icon     = "📊"

[users.analista1.permissions]
entrenar_modelos      = true
ver_metricas_tecnicas = true
ver_predicciones      = true
exportar              = true
gestionar_usuarios    = false
ver_grid_search       = true
ver_acf_pacf          = true
ver_ingresos          = true   # acceso a Proyección de Ingresos

# ── Rol Financiero — solo vista financiera ─────────────────────────────────────
[users.financiero1]
email    = "financiero@tudominio.com"
role     = "financiero"
name     = "Analista Financiero"
icon     = "💰"

[users.financiero1.permissions]
entrenar_modelos      = false
ver_metricas_tecnicas = false
ver_predicciones      = true
exportar              = true
gestionar_usuarios    = false
ver_grid_search       = false
ver_acf_pacf          = false
ver_ingresos          = true   # acceso a Proyección de Ingresos
# ──────────────────────────────────────────────────────────────────────────────

[users.gerente]
email    = "gerente@tudominio.com"
role     = "manager"
name     = "Nombre Gerente"
icon     = "💼"

[users.gerente.permissions]
entrenar_modelos      = false
ver_metricas_tecnicas = false
ver_predicciones      = true
exportar              = true
gestionar_usuarios    = false
ver_grid_search       = false
ver_acf_pacf          = false
ver_ingresos          = false  # sin acceso a Proyección de Ingresos

[users.consultor]
email    = "consultor@tudominio.com"
role     = "viewer"
name     = "Nombre Consultor"
icon     = "👁️"

[users.consultor.permissions]
entrenar_modelos      = false
ver_metricas_tecnicas = false
ver_predicciones      = true
exportar              = false
gestionar_usuarios    = false
ver_grid_search       = false
ver_acf_pacf          = false
ver_ingresos          = false  # sin acceso a Proyección de Ingresos
```

> **Importante:** `.streamlit/secrets.toml` está en `.gitignore` y **nunca debe subirse al repositorio**.

**Fallback local — hash SHA-256 (opcional):**

Si quieres mantener credenciales locales de respaldo:

```python
import hashlib
print(hashlib.sha256("mi_contraseña".encode()).hexdigest())
# → "5e884898da28047151d0e56f8dc6292773..."
```

El sistema detecta automáticamente si el valor es un hash o texto plano.

---

### 5. Arrancar en local

```bash
streamlit run app_principal.py
```

La aplicación estará disponible en `http://localhost:8501`.

---

## Despliegue en Streamlit Cloud

### 1. Conectar el repositorio

1. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
2. Haz clic en **New app**.
3. Selecciona el repositorio y la rama (`main`).
4. Entry point: `app_principal.py`.
5. Haz clic en **Deploy**.

### 2. Configurar secretos en Streamlit Cloud

1. En la app desplegada → **⋮ → Settings → Secrets**
2. Pega el contenido completo de tu `secrets.toml`
3. Guarda — la app se reiniciará automáticamente

### 3. Actualizar la aplicación

Cada `git push` a `main` dispara un redeploy automático en Streamlit Cloud (~1 min).

```bash
git add pages/1_Entrenamiento.py
git commit -m "fix: corregir umbral de descarte en Optuna"
git push origin main
```

---

## Estructura del bucket tras el primer entrenamiento

```
modelos-ml/
├── latest.txt                          ← "20260417_143000" (backup del run activo)
├── training_log.json                   ← historial JSON (backup; primario es la DB)
└── 20260417_143000/
    ├── metricas_mejoradas.json
    ├── prediccion_total_mejorada.xlsx
    ├── grid_search_results.xlsx
    ├── walk_forward_validation.xlsx
    ├── historico_total_mejorado.xlsx
    ├── modelo_total_mejorado.pkl.gz
    ├── acf_plot.png
    ├── pacf_plot.png
    └── llm_cache.json
```

---

## Resolución de problemas frecuentes

### Login falla con "Credenciales incorrectas"

**Causa probable A:** El usuario no tiene una cuenta en Supabase Auth con ese email.
**Solución:** Crear la cuenta en Supabase → Authentication → Users, o añadir el campo `password` en `secrets.toml` como fallback temporal.

**Causa probable B:** El email en `secrets.toml` no coincide con el registrado en Supabase Auth.
**Solución:** Verificar que `users.xxx.email` en `secrets.toml` es exactamente el mismo que en Supabase Auth.

---

### El modelo no carga en el Dashboard

**Causa probable:** La tabla `training_runs` está vacía (runs previos solo en `training_log.json`) o el run no tiene artefactos en Storage.

**Solución:** Lanzar un entrenamiento completo y aprobarlo. Si hay runs históricos en JSON que no aparecen en la tabla, el sistema los listará vía fallback pero no los mostrará en el selector hasta que se realice un nuevo entrenamiento.

---

### Error al subir el modelo a Supabase (413 Payload Too Large)

**Causa:** El modelo SARIMAX serializado supera el límite de la API de Supabase (~6 MB).

**Solución ya implementada:** el sistema comprime con `gzip` antes de subir (`modelo_total_mejorado.pkl.gz`). Si el error persiste, revisar la versión del cliente `supabase-py`.

---

### Las tablas `training_runs` o `audit_log` no existen

**Síntoma:** warnings en los logs `"No se pudo guardar en DB"` o `"No se pudo escribir audit_log"`.

**Solución:** Ejecutar el SQL de creación de tablas en **Supabase → SQL Editor** (ver sección 3.2 de esta guía).

---

### La sesión expira muy rápido

**Causa:** Timeout configurado a 30 minutos en `auth_system.py`.

**Solución:** Modificar `SESSION_TIMEOUT` en `core/auth_system.py`:

```python
SESSION_TIMEOUT = 60  # minutos
```

---

### El Asistente IA no responde

**Causa probable:** `GENAI_API_KEY` no está configurada o ha expirado.

**Verificación:**
```python
import streamlit as st
print(st.secrets.get("GENAI_API_KEY", "NO CONFIGURADA"))
```

Si devuelve `"NO CONFIGURADA"`, añade la clave en los secretos de Streamlit Cloud (o en `secrets.toml` local).

---

### Prophet no se instala correctamente

**Solución en macOS:**
```bash
xcode-select --install
pip install pystan==2.19.1.1
pip install prophet
```

**Solución en Linux:**
```bash
sudo apt-get install build-essential
pip install prophet
```

---

## Seguridad en producción

- Activa **Supabase Auth** para todos los usuarios — es más seguro que contraseñas en `secrets.toml`.
- Rota las claves de Supabase (anon key) y Gemini periódicamente.
- Revisa `audit_log` regularmente para detectar accesos no autorizados.
- El bucket debe ser **privado** — nunca público.
- Activa la autenticación de dos factores (2FA) en las cuentas de Streamlit Cloud y Supabase.
- Usa HTTPS en todo momento (garantizado por Streamlit Cloud y Supabase).
