# Guía de Despliegue

Esta guía explica cómo configurar el sistema en local para desarrollo y cómo desplegarlo en producción con **Streamlit Cloud**. No se incluyen credenciales reales — consulta al administrador del sistema para obtenerlas.

---

## Requisitos previos

- Python 3.10 o superior
- Cuenta en [Supabase](https://supabase.com) con un bucket de almacenamiento creado
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
supabase        — Cliente Supabase Storage
google-genai    — Cliente Google Gemini (Asistente IA)
openpyxl        — Lectura/escritura de archivos Excel
pillow          — Procesamiento de imágenes (ACF/PACF)
```

> **Nota sobre Prophet:** en algunos sistemas es necesario instalar `pystan` o `cmdstanpy` antes de Prophet. Consulta la [documentación oficial](https://facebook.github.io/prophet/docs/installation.html) si la instalación falla.

### 3. Configurar credenciales

Copia la plantilla de configuración:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edita `.streamlit/secrets.toml` con los valores reales. La estructura es:

```toml
# API de Google Gemini (Asistente IA)
GENAI_API_KEY = "tu_clave_gemini_aqui"

[supabase]
url    = "https://<tu-proyecto>.supabase.co"
key    = "tu_anon_key_de_supabase"
bucket = "modelos-ml"

# Usuarios del sistema — añade tantos como necesites
[users.admin]
password    = "contraseña_segura"   # o hash SHA-256
role        = "admin"
name        = "Nombre Completo"
icon        = "👑"

[users.admin.permissions]
entrenar_modelos = true
ver_dashboard    = true
exportar         = true

[users.analista1]
password = "otra_contraseña"
role     = "analyst"
name     = "Analista Uno"
icon     = "📊"

[users.analista1.permissions]
entrenar_modelos = true
ver_dashboard    = true
exportar         = true

[users.gerente]
password = "contraseña_gerente"
role     = "manager"
name     = "Nombre Gerente"
icon     = "💼"

[users.gerente.permissions]
ver_dashboard = true
exportar      = false

[users.consultor]
password = "contraseña_viewer"
role     = "viewer"
name     = "Nombre Consultor"
icon     = "👁️"

[users.consultor.permissions]
ver_dashboard = true
```

> **Importante:** el fichero `.streamlit/secrets.toml` está en `.gitignore` y **nunca debe subirse al repositorio**. Contiene credenciales sensibles.

**Contraseñas con hash SHA-256 (opcional):**

Si prefieres no almacenar contraseñas en texto plano, puedes usar el hash SHA-256:

```python
import hashlib
print(hashlib.sha256("mi_contraseña".encode()).hexdigest())
# → "5e884898da28047151d0e56f8dc6292773..."
```

El sistema detecta automáticamente si el valor es un hash o texto plano.

### 4. Configurar Supabase Storage

En tu proyecto de Supabase:

1. Ve a **Storage → Buckets → New bucket**.
2. Nombre: `modelos-ml` (debe coincidir con `bucket` en `secrets.toml`).
3. Visibilidad: **privado** (Private).
4. No es necesario configurar políticas adicionales — el acceso se hace con la `anon key` del SDK.

La primera vez que se lance un entrenamiento, el sistema creará automáticamente los ficheros `latest.txt` y `training_log.json` en el bucket.

### 5. Arrancar en local

```bash
streamlit run app_principal.py
```

La aplicación estará disponible en `http://localhost:8501`.

---

## Despliegue en Streamlit Cloud

Streamlit Cloud permite desplegar aplicaciones Streamlit directamente desde GitHub, de forma gratuita para repositorios públicos o con plan de pago para privados.

### 1. Conectar el repositorio

1. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
2. Haz clic en **New app**.
3. Selecciona el repositorio y la rama (`main`).
4. Entry point: `app_principal.py`.
5. Haz clic en **Deploy**.

### 2. Configurar secretos en Streamlit Cloud

Los secretos **no se copian desde el repositorio** — deben introducirse manualmente en la plataforma:

1. En la app desplegada, ve a **⋮ → Settings → Secrets**.
2. Pega el contenido completo de tu `secrets.toml` (sin los comentarios si lo prefieres).
3. Guarda los cambios — la app se reiniciará automáticamente.

### 3. Actualizar la aplicación

Cada `git push` a la rama `main` dispara automáticamente un redeploy en Streamlit Cloud. No es necesario ningún paso manual adicional.

```bash
git add pages/1_Entrenamiento.py
git commit -m "fix: corregir umbral de descarte en Optuna"
git push origin main
# → Streamlit Cloud detecta el push y redespliega en ~1 min
```

---

## Estructura del bucket Supabase tras el primer entrenamiento

```
modelos-ml/
├── latest.txt                          ← "20260325_143000" (run activo)
├── training_log.json                   ← historial de todos los runs
└── 20260325_143000/
    ├── metricas_mejoradas.json         ← parámetros, AIC, MAPE, configuración
    ├── prediccion_total_mejorada.xlsx  ← predicciones futuras con IC
    ├── grid_search_results.xlsx        ← resultados de todos los trials Optuna
    ├── walk_forward_validation.xlsx    ← real vs predicho por mes
    ├── historico_total_mejorado.xlsx   ← serie temporal histórica
    ├── modelo_total_mejorado.pkl.gz    ← objeto SARIMAX serializado (comprimido)
    ├── acf_plot.png                    ← gráfico de autocorrelación
    └── pacf_plot.png                   ← gráfico de autocorrelación parcial
```

Cada nuevo entrenamiento crea una carpeta nueva con el timestamp como nombre. Los runs anteriores nunca se borran automáticamente.

---

## Resolución de problemas frecuentes

### El modelo no carga en el Dashboard

**Causa probable:** `latest.txt` no existe en el bucket o apunta a un run que no tiene todos los archivos.

**Solución:** Lanzar un entrenamiento completo desde la pestaña Entrenamiento y aprobarlo. Esto crea o actualiza `latest.txt`.

---

### Error al subir el modelo a Supabase (413 Payload Too Large)

**Causa:** El modelo SARIMAX serializado supera el límite de tamaño de la API de Supabase (~6 MB).

**Solución ya implementada:** el sistema comprime el modelo con `gzip` antes de subirlo (`modelo_total_mejorado.pkl.gz`). Si el error persiste, revisar si hay una versión del cliente `supabase-py` que tenga un límite diferente.

---

### Prophet no se instala correctamente

**Causa:** Prophet requiere `pystan` como compilador de Stan, que en algunos sistemas necesita un compilador C++ instalado.

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

### La sesión expira muy rápido

**Causa:** El timeout está configurado a 30 minutos por defecto en `auth_system.py`.

**Solución:** Modificar la constante `SESSION_TIMEOUT` en `auth_system.py`:

```python
SESSION_TIMEOUT = 60  # minutos (antes era 30)
```

---

### El Asistente IA no responde

**Causa probable:** La clave `GENAI_API_KEY` no está configurada en `secrets.toml` o ha expirado.

**Verificación:**
```python
# En una celda de notebook o script temporal:
import streamlit as st
print(st.secrets.get("GENAI_API_KEY", "NO CONFIGURADA"))
```

Si devuelve `"NO CONFIGURADA"`, añade la clave en los secretos de Streamlit Cloud (o en el `secrets.toml` local).

---

## Variables de entorno opcionales

Streamlit Cloud permite pasar variables de entorno adicionales desde la sección Secrets usando la sintaxis de TOML estándar. No se requieren variables de entorno adicionales para el funcionamiento base del sistema.

---

## Seguridad en producción

- Usa contraseñas con hash SHA-256 para todos los usuarios.
- Rota las claves de Supabase y Gemini periódicamente.
- Revisa los logs de acceso de Supabase para detectar accesos no autorizados al bucket.
- El bucket debe ser **privado** — nunca público.
- Activa la autenticación de dos factores en las cuentas de Streamlit Cloud y Supabase.
