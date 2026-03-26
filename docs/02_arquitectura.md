# Arquitectura del Sistema

## Visión general

El sistema sigue una arquitectura **modular de tres capas**: interfaz (Streamlit), lógica de negocio (Python) y almacenamiento persistente (Supabase Storage). No existe base de datos relacional: todos los artefactos del modelo se serializan en ficheros y se almacenan en un bucket de objetos en la nube.

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT CLOUD                      │
│                                                         │
│  app_principal.py  ←── Entry point / autenticación     │
│       │                                                 │
│       ├── pages/1_Entrenamiento.py                      │
│       ├── pages/2_Dashboard.py                          │
│       └── pages/3_Prophet_vs_SARIMA.py                  │
│                                                         │
│  Módulos compartidos:                                   │
│  ├── auth_system.py       (sesiones y permisos)         │
│  ├── supabase_io.py       (capa de I/O centralizada)    │
│  └── utils_validacion.py  (validación de datos)         │
└───────────────────┬─────────────────────────────────────┘
                    │ HTTPS / supabase-py SDK
┌───────────────────▼─────────────────────────────────────┐
│              SUPABASE STORAGE (bucket: modelos-ml)      │
│                                                         │
│  latest.txt                ← apunta al run activo       │
│  training_log.json         ← historial de runs          │
│  20260325_143000/          ← carpeta por run            │
│    ├── metricas_mejoradas.json                          │
│    ├── prediccion_total_mejorada.xlsx                   │
│    ├── grid_search_results.xlsx          (Optuna)       │
│    ├── walk_forward_validation.xlsx                     │
│    ├── historico_total_mejorado.xlsx                    │
│    ├── modelo_total_mejorado.pkl.gz      (comprimido)   │
│    ├── acf_plot.png                                     │
│    └── pacf_plot.png                                    │
└─────────────────────────────────────────────────────────┘
```

> No se persiste ningún dato en disco local. Todo el almacenamiento es en la nube, lo que permite que la aplicación sea **stateless** y se pueda escalar o reiniciar sin pérdida de información.

---

## Módulos del sistema

### `app_principal.py` — Entry point

Es la página de inicio de la aplicación Streamlit. Su responsabilidad es:
1. Verificar si el usuario está autenticado (delegando en `auth_system`).
2. Si no lo está, mostrar el formulario de login.
3. Si lo está, presentar la página de bienvenida con accesos directos a las secciones según el rol.

Streamlit detecta automáticamente los archivos en `/pages/` y construye la navegación lateral.

---

### `auth_system.py` — Autenticación y permisos

Gestiona el ciclo de vida de la sesión de usuario. Las credenciales se leen de `st.secrets` (fichero `secrets.toml` en local, variable de entorno en producción) y **nunca se almacenan en el código fuente**.

**Funciones principales:**

| Función | Descripción |
|---------|-------------|
| `init_session_state()` | Inicializa variables de sesión: `authenticated`, `username`, `role`, `permissions` |
| `show_login_page(title)` | Renderiza el formulario de login con logo corporativo |
| `check_session_timeout()` | Invalida la sesión si supera 30 minutos de inactividad |
| `has_permission(name)` | Retorna `True/False` para un permiso específico |
| `show_header(title, subtitle)` | Renderiza el encabezado corporativo en todas las páginas |

**Ejemplo de verificación de permiso:**
```python
# En cualquier página, antes de mostrar funcionalidad sensible:
if not has_permission('entrenar_modelos'):
    st.error("❌ No tienes permiso para acceder a esta sección")
    st.stop()
```

**Permisos disponibles:**

| Permiso | Descripción |
|---------|-------------|
| `entrenar_modelos` | Acceso a Entrenamiento y Prophet vs SARIMA |
| `ver_dashboard` | Acceso al Dashboard |
| `exportar` | Descarga de predicciones en CSV |

---

### `supabase_io.py` — Capa de I/O centralizada

Toda la comunicación con Supabase Storage pasa por este módulo. Ninguna página importa el cliente de Supabase directamente; esto centraliza el manejo de errores y facilita cambiar el backend de almacenamiento sin tocar las páginas.

**Funciones principales:**

| Función | Entrada | Salida |
|---------|---------|--------|
| `save_to_dashboard(run_name, model, pred, grid, wf, hist, metricas, fig_acf, fig_pacf)` | Todos los artefactos del entrenamiento | — (sube a Supabase) |
| `load_precargados(run_name)` | Nombre del run (`'20260325_143000'`) | `(metricas, pred_df, grid_df, wf_df, hist_series)` |
| `get_available_runs()` | — | Lista de runs disponibles |
| `get_default_run()` | — | Nombre del run activo (`latest.txt`) |
| `approve_model(run_name)` | Nombre del run | Actualiza `latest.txt` |
| `save_training_log(entry)` | Dict con métricas del run | Añade entrada a `training_log.json` |
| `load_training_log()` | — | Lista de dicts con historial |

**Ejemplo de carga de un modelo:**
```python
import supabase_io as sio

run = sio.get_default_run()                          # '20260325_143000'
metricas, pred, grid, wf, hist = sio.load_precargados(run)

print(metricas['mejor_modelo']['order'])             # (1, 1, 1)
print(metricas['walk_forward_validation']['mape'])   # 5.34
print(pred.head())
#         Fecha       Mes  Predicción  IC_Inferior  IC_Superior
# 0  2026-03-31  marzo 2026        47.2         38.1         56.3
```

**Estructura del objeto `metricas`:**
```json
{
  "fecha_entrenamiento": "20260325_143000",
  "usuario": "analista01",
  "configuracion": {
    "modelo_filtro": "TIGGO 2",
    "horizonte": 6,
    "max_ventas": 100
  },
  "datos_limpios": {
    "meses_datos": 62,
    "periodo": "2021-01 a 2026-02"
  },
  "mejor_modelo": {
    "order": [1, 1, 1],
    "seasonal_order": [1, 1, 1, 12],
    "aic": 450.23,
    "combinaciones_validas": 67,
    "combinaciones_descartadas": 13
  },
  "walk_forward_validation": {
    "mape": 5.34,
    "meses_evaluados": 6
  },
  "predicciones_futuras": {
    "proximo_mes": 45.2
  }
}
```

---

### `utils_validacion.py` — Validación de datos

Valida la calidad del Excel de ventas antes de lanzar el entrenamiento. Los checks se ejecutan automáticamente en la pestaña **Validación**.

**Checks aplicados:**

| Check | Umbral | Acción si falla |
|-------|--------|-----------------|
| Columnas requeridas (`FECHA-VENTA`, `MODELO3`, `MARCA`) | — | Error bloqueante |
| Fechas parseables | Máx. 5% inválidas | Error bloqueante |
| Período mínimo | ≥ 36 meses | Error bloqueante |
| Datos faltantes por columna | Máx. 5% | Advertencia |
| Outliers (regla 3σ) | — | Advertencia |

---

## Flujo de datos: Entrenamiento → Dashboard

El siguiente diagrama muestra cómo los datos fluyen desde la carga del Excel hasta que el gerente consulta una predicción:

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
  └── Verifica columnas, fechas, período mínimo, outliers
      │
      ▼
[Tab 4: Entrenamiento]
  ├── Filtra por marca + modelo + rango de fechas
  ├── Resample mensual → Serie temporal
  ├── Construye variable exógena (ventas otros modelos)
  ├── Test ADF (estacionariedad)
  ├── Optuna TPE (80 trials) → mejores parámetros SARIMA
  ├── Walk-forward validation (últimos N meses)
  ├── Modelo final + forecast 6 meses
  └── Sube artefactos a Supabase (run_name = timestamp)
      │
      ▼
[Tab 5: Comparación]
  └── Compara MAPE/AIC nuevo vs activo
      │ si "Aprobar"
      ▼
  latest.txt ← run_name
      │
      ▼
[Dashboard]
  └── Lee latest.txt → carga artefactos → muestra predicciones
```

---

## Gestión de versiones de modelos

Cada ejecución de entrenamiento genera un **run** identificado por un timestamp (`YYYYMMDD_HHMMSS`). Los runs se acumulan en Supabase y nunca se borran automáticamente.

El fichero `latest.txt` actúa como **puntero al modelo de producción activo**. Solo se actualiza cuando el analista hace clic en "Aprobar" en la pestaña Comparación.

Esto permite:
- Revertir a una versión anterior cambiando `latest.txt`
- Consultar cualquier run histórico desde el Dashboard sin afectar producción
- Auditar quién entrenó cada modelo y cuándo (campo `usuario` en `metricas`)

---

## Consideraciones de seguridad

- Las credenciales (Supabase, Gemini API, usuarios) se almacenan exclusivamente en `st.secrets`, nunca en el código fuente ni en el repositorio.
- El fichero `.streamlit/secrets.toml` está listado en `.gitignore`.
- Las contraseñas de usuarios pueden almacenarse como texto plano (entornos internos) o como hash SHA-256.
- El timeout de sesión (30 min) limita la exposición de sesiones inactivas.
- El sistema no expone datos de ventas individualmente — solo series temporales agregadas por mes.
