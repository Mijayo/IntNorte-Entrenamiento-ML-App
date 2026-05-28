# Changelog — Sistema TIGGO 2

Todas las versiones relevantes del proyecto, de más reciente a más antigua.

---

### 2026-05-28 (v23)

- **config(auth)**: `SESSION_TIMEOUT` aumentado de 30 → **60 minutos** en `core/auth_system.py`. Reduce interrupciones de sesión para flujos de trabajo de entrenamiento largos.
- **docs**: README, CHANGELOG, `docs/02_arquitectura.md`, `docs/03_guia_usuario.md` y `docs/05_despliegue.md` actualizados para reflejar el nuevo timeout de sesión.

---

### 2026-05-27 (v22)

- **refactor(comparativa + entrenamiento)**: Unificación total — ambas páginas usan ahora **SARIMAX** (con variable exógena `ventas_otros`) idéntico al modelo de producción. Antes, Comparativa ML ejecutaba SARIMA *sin* exógena, produciendo una comparación no equitativa que inflaba el MAPE del modelo de serie temporal frente a los algoritmos ML supervisados.
- **feat(supabase_io)**: `save_to_dashboard` guarda `historico_exog.xlsx` cuando el run incluye variable exógena. `load_precargados` pasa de devolver 5 valores a **6** (añade la serie exógena como último elemento). Retrocompatible: devuelve `None` para runs anteriores sin el archivo.
- **feat(comparativa)**: Los parámetros SARIMAX (p, d, q, P, D, Q) se auto-completan desde `metricas_mejoradas.json` del run cargado — ya no es necesario introducirlos manualmente.
- **feat(comparativa)**: Al cargar un run se muestra el estado de la variable exógena (Pearson r, disponible / no disponible) con badge de aviso para runs históricos sin exog.
- **refactor(2_Dashboard, 3_Proyeccion_Ingresos)**: Desempaquetado de `load_precargados` actualizado a 6-tupla (`_exog` ignorado en estas páginas).
- **chore**: Renombrado de cadenas literales `"SARIMA"` → `"SARIMAX"` en colores, checkboxes, intérprete de resultados, sección de publicación y mensajes informativos de `4_Comparativa_ML.py`.

---

### 2026-05-27 (v21)

- **feat(auth)**: Nuevo rol **`financiero`** — usuario `financiero` con icono 💰, badge propio en sidebar y contraseña configurable en `secrets.toml`.
- **feat(rbac)**: Nuevo permiso `ver_ingresos` — controla el acceso a la página **Proyección de Ingresos**. Habilitado para `admin`, `analyst` y el nuevo rol `financiero`; deshabilitado para `manager` y `viewer`.
- **refactor(3_Proyeccion_Ingresos)**: Guard de acceso `has_permission('ver_ingresos')` añadido tras el check de autenticación. La página ya no es accesible a todos los roles — muestra `🔒 Acceso restringido` y detiene la ejecución para roles sin permiso.
- **feat(pages)**: **`pages/3_Proyeccion_Ingresos.py`** extraída del tab 3 de `2_Dashboard.py` como página independiente con navegación propia.
- **docs**: README, CHANGELOG, `docs/03_guia_usuario.md` y `docs/05_despliegue.md` actualizados para reflejar el nuevo rol, el nuevo permiso y las tablas de acceso revisadas.
- **chore(secrets)**: `secrets.toml` y `secrets.toml.example` actualizados: `ver_ingresos` añadido a todos los usuarios existentes + bloque `[users.financiero]` nuevo.

---

### 2026-05-10 (v20)
- **feat(ui)**: Rediseño visual completo — línea gráfica alineada con la presentación Interamericana Norte / SAA 2025
- **refactor(styles)**: Paleta global migrada de electric cyan (`#00E0FF`) a naranja vibrante (`#FF4800`) como acento primario; lime-yellow (`#C2FF00`) como secundario; fondos cálidos oscuros (`#0D0C0A`)
- **feat(home)**: Nueva sección hero en `app_principal.py` con breadcrumb `INTERAMERICANA / NORTE · FILE 2025/SAA · CHERY TIGGO 2`, headline de display grande con acento naranja y subtítulo con orden SARIMA
- **feat(home)**: Feature cards rediseñadas con KPI inline por card: `SARIMA` (Entrenamiento), `14.65 %` MAPE (Dashboard), `5 modelos` (Comparativa ML)
- **feat(home)**: Footer `EQUIPO · PROG · CLNT` con notación estilo slides de presentación
- **refactor(styles)**: Card accent bars actualizadas — azul→naranja, verde→lime, amber→violeta

---

### 2026-05-04 (v19)
- **feat**: **Tab dedicado "💰 Proyección Ingresos"** — nueva pestaña disponible para todos los roles (viewer, manager, admin, analyst) en posición tabs[2]. Inputs: precio unitario (USD $), margen neto (%) y tipo de cambio para conversión a moneda local. KPIs: unidades totales, ingresos centrales, rango IC 95% en dólares, beneficio neto (si margen > 0). Gráfico de barras con banda IC 95% en overlay y línea de beneficio neto. Tabla mensual con fila de totales y gradiente de color. Exportar CSV para roles con permiso `exportar`.
- **refactor**: La sección de proyección financiera que existía al fondo del tab Predicciones fue eliminada — el tab dedicado la reemplaza con mayor detalle y formato USD.
- **refactor**: Todos los índices de tabs de admin/analyst desplazados +1 (ACF/PACF → 3, Grid Search → 4, Walk-Forward → 5, Métricas → 6, Asistente IA → 7, Concesionarios → 8). Manager: Recomendaciones → 3, Asistente IA → 4, Concesionarios → 5.

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
