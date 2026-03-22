# Sistema TIGGO 2 — Predicción de Ventas con SARIMA

Sistema multipage en Streamlit para entrenar modelos SARIMA y visualizar predicciones de demanda de vehículos. Desplegado en **Streamlit Cloud** con almacenamiento persistente en **Supabase Storage** y control de acceso por roles.

---

## Arquitectura

```
app_principal.py          ← Entry point (autenticación + página de inicio)
pages/
├── 1_Entrenamiento.py    ← Entrenamiento SARIMA (Admin / Analista)
└── 2_Dashboard.py        ← Dashboard de negocio (todos los roles)
auth_system.py            ← Autenticación, sesiones y show_header()
supabase_io.py            ← Capa de I/O centralizada (Supabase Storage)
utils_validacion.py       ← Validación de datos de entrada
requirements.txt
.streamlit/
├── secrets.toml          ← Credenciales reales (NO en el repo)
└── secrets.toml.example  ← Plantilla (sí en el repo)
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
matplotlib, plotly, pillow, openpyxl, supabase
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

| Rol | Entrenamiento | Tabs del Dashboard |
|-----|:-------------:|---------------------|
| `admin` | ✅ | Dashboard, Predicciones, ACF/PACF, Grid Search, Walk-Forward, Métricas técnicas |
| `analyst` | ✅ | Dashboard, Predicciones, ACF/PACF, Grid Search, Walk-Forward, Métricas técnicas |
| `manager` | — | Dashboard, Predicciones, Recomendaciones de compra |
| `viewer` | — | Dashboard, Predicciones |

---

## Flujo de trabajo

```
1. [Entrenamiento]  Cargar uno o varios Excel con datos de ventas
2. [Entrenamiento]  Validación automática de calidad de datos
3. [Entrenamiento]  Test ADF de estacionariedad
4. [Entrenamiento]  Grid search → mejores parámetros SARIMA (AIC + MAPE)
5. [Entrenamiento]  Walk-forward validation sobre los últimos N meses
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

Excel (`.xlsx` / `.xls`) con hoja `Hoja1`:

| Columna | Descripción |
|---------|-------------|
| `FECHA-VENTA` | Fecha de venta (parseable por pandas) |
| `MARCA` | Marca del vehículo |
| `MODELO3` | Nombre del modelo |

### Parámetros configurables (UI)

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| Marca | `CHERY` | Filtro de marca |
| Modelo | `TIGGO 2` | Filtro de modelo |
| Fecha inicio | `2021-01-01` | Ignorar ventas anteriores |
| Horizonte | `6` meses | Meses a predecir (3–12) |
| Máx. ventas | `100` unid./mes | Límite superior de predicciones válidas |
| Excluir mes actual | `true` | Eliminar mes incompleto del entrenamiento |

### Modelo SARIMA

- **Algoritmo**: SARIMAX con variable exógena (ventas de otros modelos de la misma marca)
- **Espacio de búsqueda**: `p ∈ {1,2,3,5,7}`, `d=1`, `q ∈ {0,1,2}`, estacional `(P=1, D=1, Q∈{0,1,2}, m=12)`
- **Criterio**: AIC más bajo con MAPE < 100% y predicciones dentro del rango configurado
- **Intervalos de confianza**: 95% en todos los puntos del forecast

### Flujo de aprobación

La pestaña **Comparación** muestra métricas lado a lado con el modelo de producción (MAPE, AIC, próximo mes). Al hacer clic en **Aprobar** se actualiza `latest.txt` en Supabase y el Dashboard refleja el cambio inmediatamente. Los runs no aprobados quedan en el historial sin afectar producción.

---

## App 2 — Dashboard (`pages/2_Dashboard.py`)

### Selector de versión

La barra lateral lista todos los runs disponibles ordenados por fecha. El run activo (apuntado por `latest.txt`) aparece con 🟢. Los históricos con 🔵 pueden seleccionarse sin alterar producción.

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

## Licencia

Uso interno. No distribuir públicamente.
