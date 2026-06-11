# Changelog — Sistema TIGGO 2

Todas las versiones relevantes del proyecto, de más reciente a más antigua.

---

### 2026-06-11 (v2) — i18n español + contexto cadena de suministro en asistente IA

#### `pages/3_Dashboard.py`

- **fix(i18n)**: Todos los nombres de mes en el Dashboard ahora se muestran en español. Causa raíz: `strftime('%B %Y')` y `strftime('%b %Y')` usan la locale del sistema operativo (inglés en macOS/Linux), produciendo "July 2026" en lugar de "Julio 2026". Fix: añadidos helpers `_MESES_ES`, `_traducir_mes()` y `_mes_es()` al inicio del módulo; todos los `strftime('%b')` y `strftime('%B')` en la sección de cadena de suministro y en las tablas walk-forward (Tab Predicciones y Tab Walk-Forward) reemplazados por indexación directa en `_MESES_ES[dt.month - 1]`.
- **feat(asistente_ia)**: El contexto enviado al LLM (Gemini 2.5 Flash) ahora incluye los datos de cadena de suministro Chery: lead times (mín. 15d · promedio 23d · máx. 30d), términos de financiación (0% interés primeros 60 días, 8% anual desde el día 61), ruta logística (Callao → Lima → Piura/Chiclayo → Tarapoto/Cajamarca) y las ventanas de pedido calculadas dinámicamente para cada mes del horizonte de predicción (fechas conservadora, óptima y agresiva). Antes, preguntas como "¿Cuándo tengo que hacer el pedido para julio?" recibían una respuesta incorrecta indicando que la información estaba fuera de alcance.

#### `pages/1_Entrenamiento.py`

- **fix(i18n)**: La columna `'Mes'` de `predicciones` ahora se genera en español (`'Julio 2026'` en lugar de `'July 2026'`) tanto al mostrar en pantalla como al guardar en Supabase. Fix aplicado en la construcción del DataFrame de predicciones futuras (línea que usaba `fechas_futuras.strftime('%B %Y')`).

---

### 2026-06-11 — Optimización de caché (perf)

#### `core/supabase_io.py`

- **perf(cache)**: `load_llm_cache(run_name)` ahora lleva `@st.cache_data(ttl=3600)`. Evita descargar el JSON de caché Gemini en cada rerender; se invalida desde `save_llm_cache()`.
- **perf(cache)**: `load_training_log()` ahora lleva `@st.cache_data(ttl=300)`. Evita queries DB repetidas al historial de entrenamientos; se invalida desde `save_training_log()`.
- **perf(cache)**: `get_audit_log(limit)` ahora lleva `@st.cache_data(ttl=60)`. Evita queries repetidas al audit log en la página de Administración.
- **perf(invalidación)**: `save_to_dashboard()` y `approve_model()` reemplazan `st.cache_data.clear()` global por invalidación granular de 6 funciones (`get_available_runs`, `get_default_run`, `load_precargados`, `load_current_model`, `get_runs_df`, `load_training_log`). Conserva cachés no relacionados (audit log, imágenes ACF/PACF, caché LLM) y reduce la latencia del primer rerender tras guardar o aprobar un modelo.

#### `pages/3_Dashboard.py`

- **perf(cache)**: Contexto LLM extraído a `_build_llm_context(run_name)` (`@st.cache_data ttl=300`). Reutiliza el string entre rerenders sin recalcular métricas ni hacer llamadas a `load_precargados` de forma redundante.
- **perf(cache)**: Figura walk-forward extraída a `_build_wf_figure(run_name)` (`@st.cache_data ttl=600`). El objeto `go.Figure` se construye una sola vez por run y se reutiliza en rerenders y navegación de tabs.

---

### 2026-06-10 (hotfix #2)

- **ux(concesionarios/mapa)**: Zoom inicial del mapa de Perú aumentado de `4.5` → `5.5` para mostrar los concesionarios más próximos al visualizar la página.
- **feat(concesionarios/mapa)**: Controlador de zoom nativo añadido al mapa. La barra de herramientas del mapa muestra tres botones (**+**, **−**, **reset vista**) en la esquina superior derecha mediante `modeBarButtonsToKeep: ['zoomInMapbox', 'zoomOutMapbox', 'resetViewMapbox']`. El scroll con ratón y el drag para desplazar el mapa siguen operativos.

---

### 2026-06-10

- **feat(concesionarios/mapa)**: Mapa geográfico interactivo de Perú añadido al tab **📊 Resumen** de `pages/4_Concesionarios.py`, entre el gráfico de barras de ventas totales y la distribución de modelos.
  - Renderizado con `px.scatter_mapbox` sobre tiles **OpenStreetMap** (sin API key, funciona en local y Streamlit Cloud).
  - Burbujas con **tamaño proporcional a ventas** y **color por concesionario** (misma paleta `COLORS['series']`).
  - Tooltip: nombre del concesionario, unidades vendidas y % del total.
  - Matching automático nombre → coordenadas: busca el nombre de ciudad dentro del nombre del concesionario. Ciudades incluidas: Lima, Callao, Piura, Chiclayo, Tarapoto, Cajamarca, Trujillo, Arequipa, Cusco, Iquitos, Huancayo, Puno.
  - Si ningún concesionario contiene una ciudad reconocible, se muestra un mensaje de ayuda descriptivo en lugar del mapa.
  - Diccionario `_COORDS_PERU` y función `_coords_concesionario()` añadidos como utilidades al inicio del módulo.

---

### 2026-06-06 (hotfix #4)

- **ux(dashboard/vs_descartados)**: La tabla "¿Por qué se descartaron las otras familias?" reemplaza `st.dataframe` por una tabla HTML personalizada con estilos CSS inline. Mejoras: (1) fila SARIMA destacada en verde con badge **✓ SELECCIONADO**, filas descartadas en gris oscuro con badge **✗ DESCARTADO**; (2) columnas "Por qué ganó" / "Por qué no" unificadas en una sola columna "Veredicto" con texto completo sin truncar; (3) badges de familia con color propio (azul violeta) para distinguir visualmente "Serie Temporal" de "ML — lag features"; (4) tags secundarios (`(p,d,q)(P,D,Q)[12]`, `Pearson`, `lag_12`) en azul tenue para desambiguar sin saturar; (5) hover brightness para facilitar la lectura fila a fila.

---

### 2026-06-06 (hotfix #3)

- **fix(dashboard/cadena_suministro)**: La sección "🚚 ¿Cuándo hacer el pedido?" ahora ancla siempre al **mes siguiente al mes actual** en lugar del primer mes del horizonte del modelo. La lógica anterior usaba `pred_total.iloc[0]`, que depende de cuándo se entrenó el modelo — si el run fue aprobado hace varias semanas, el primer mes del forecast ya podría haber pasado o no coincidir con el mes siguiente real desde hoy. **Fix:** se detecta la fecha real con `pd.Timestamp.today()`, se calcula el inicio del mes siguiente con `pd.DateOffset(months=1)`, y se busca la predicción correspondiente en `pred_total` por período mensual. Si el mes siguiente no está en el horizonte activo (edge case), hace fallback gracioso al primer mes disponible del forecast. Los KPIs de "Días al deadline óptimo", las fechas del timeline y las unidades proyectadas se calculan ahora contra el mes correcto desde cualquier fecha de ejecución.

---

### 2026-06-06 (hotfix #2)

- **fix(dashboard/seguimiento_produccion)**: La sección "📡 Seguimiento en Producción — Real vs Predicción" siempre mostraba el mensaje *"Hay ventas reales registradas, pero no coinciden con las fechas de predicción del modelo activo"* aunque las ventas estuviesen correctamente registradas. Causa: `pred_total["Fecha"]` usa frecuencia `ME` (fin de mes, ej. `2026-04-30`), mientras que las ventas reales se guardan con fecha `2026-04-01`. El lookup por diccionario `{pd.Timestamp(row["Fecha"]): ...}` nunca encontraba coincidencia exacta. **Fix:** ambas fechas se normalizan a inicio de mes con `.to_period("M").to_timestamp()` antes del lookup, igual que lo hace `7_Registrar_Ventas.py`. El gráfico real vs predicción y la alerta de drift ahora se muestran correctamente.

---

### 2026-06-06 (hotfix)

- **fix(dashboard/cadena_suministro)**: El timeline visual de la sección "🚚 Cadena de Suministro — ¿Cuándo hacer el pedido?" aparecía como código HTML en bruto en la app. Causa: `st.markdown(..., unsafe_allow_html=True)` no renderizaba el bloque en Streamlit 1.50. **Fix:** reemplazado por `st.html()` (disponible desde Streamlit 1.31), la API dedicada para renderizar HTML arbitrario sin pasar por el procesador de Markdown.

---

### 2026-06-06 (v41)

- **feat(dashboard/recomendaciones)**: Nueva sección **"🚚 Cadena de Suministro — ¿Cuándo hacer el pedido?"** en el Tab Recomendaciones, basada en datos reales del cliente Chery 2025.
  - 3 KPI cards: lead time promedio (22–24 días), fecha de pedido óptima para el mes predicho (calculada dinámicamente desde `pred_total['Fecha'].iloc[0]`), días que quedan al deadline óptimo (con semáforo rojo/ámbar/verde).
  - Timeline visual con 4 hitos enlazados: **Conservador** (–30 días / máximo histórico), **Óptimo** (–23 días / media real 2025), **Agresivo** (–15 días / mínimo histórico), **Inicio del mes** predicho.
  - Mapa de ruta logística Chery: Puerto Callao → Almacén Lima → Piura / Chiclayo → *(opt.)* Tarapoto / Cajamarca.
  - Nota sobre la ventana libre de interés: **0% los primeros 60 días** → 8% anual a partir del día 61. Pedir cercano al inicio del mes demandado maximiza la ventana libre.
- **fix(dashboard/marco_teorico)**: Tiempo de reposición corregido de *"60–90 días (importaciones)"* → *"15–30 días (promedio Chery 2025: 22–24 días)"* en el expander 📚 Marco teórico del Tab Recomendaciones.
- **feat(dashboard/asistente_ia)**: Contexto enviado a Gemini enriquecido con datos reales de supply chain: lead time min/avg/max, ruta logística completa (Callao → Lima → Piura/Chiclayo → Tarapoto/Cajamarca), apertura de sedes dic. 2025, y tasa de inventario Chery (60 días gratis + 8% anual).
- **fix(proyeccion_ingresos)**: Default `costo_fin_pct` actualizado de **1.5% → 0.7%** mensual, reflejando la tasa real Chery: 8% anual = 0.67%/mes. Help text actualizado con la estructura de costos real (60 días libres, tasa desde día 61).
- **feat(auth)**: **Demo bypass** — cuando `demo_mode = true` en `secrets.toml`, `guard_page()` salta el login y abre sesión automáticamente como rol `admin` con nombre "Demo ISDI". Facilita demostraciones en vivo sin revelar credenciales.

---

### 2026-06-05 (docs)

- **docs**: Creados `docs/08_conclusiones_iteracion3.md` y `docs/09_conclusiones_release_final.md`. Corregida evolución MAPE en `docs/07_conclusiones_iteracion2.md`.
  - Evolución correcta: **27.89% (Iter1) → 10.32% (Iter2) → 14.65% (Iter3)**
  - Iter3: MAPE walk-forward **14.65%** · SARIMA(2,0,1)(1,0,2)\[12\] · v40 · justificación de re-entrenamiento trimestral · modelos descartados con métricas · narrativa de regresión por datos 2026.
  - Release Final: resumen ejecutivo de las 3 iteraciones, ROI cuantificado, roadmap competitivo, aprendizajes del proyecto.

---

### 2026-06-04 (v40)

- **fix(registrar_ventas)**: Tab **"📡 Comparativa en Producción"** ya no aparece vacía cuando las ventas reales registradas corresponden a meses predichos por el modelo. El problema era un mismatch de formato de fecha: `pred_total` usa `freq='ME'` (mes-fin, ej. `2026-04-30`) mientras que el formulario de registro guarda fechas como primer día del mes (`2026-04-01`). El `merge` por igualdad exacta devolvía cero filas. **Fix:** ambas columnas de fecha se normalizan a inicio de mes con `.dt.to_period("M").dt.to_timestamp()` antes del join. MAPE de producción, gráfico real vs predicción e IC 95% ahora se muestran correctamente.

---

### 2026-06-03 (v39)

- **refactor(entrenamiento)**: `_load_preloaded()` simplificada — ya solo carga `data/processed/veh_ml_features.xlsx` (o su equivalente en Supabase Storage). Eliminada la constante `_PRELOADED_STOCK` y la carga de `data/raw/Stock Vehiculos.xlsx`. La función devuelve un único `pd.DataFrame` en vez de una tupla; el fallback de Supabase extrae solo `df_v` de `sio.load_datos_precargados()`. Eliminada la línea `st.session_state["df_stock"]` del flujo precargado — el stock de vehículos es exclusivo del flujo de carga manual si el usuario sube su propio Excel.
- **refactor(concesionarios)**: `_cargar_precargado()` actualizada — el fallback local ahora apunta a `data/processed/veh_ml_features.xlsx` (hoja `Hoja1`) en lugar de `data/raw/Historico_Ventas.xlsx`. La fuente primaria (Supabase Storage) no cambia. Textos de UI actualizados para mostrar `veh_ml_features.xlsx` en el badge de datos precargados y en el mensaje de confirmación de subida a Supabase.

---

### 2026-06-03 (v38)

- **fix(concesionarios)**: `_cargar_precargado()` ahora carga `Historico_Ventas.xlsx` desde **Supabase Storage** como fuente primaria (`sio.load_historico_ventas()`), con fallback al archivo local `data/raw/Historico_Ventas.xlsx`. Elimina el `FileNotFoundError` en producción. La lógica de normalización de columnas se extrajo a `_normalizar_df()` para reutilizarla tanto en la carga precargada como en el uploader de Excel personalizado.
- **feat(concesionarios)**: Botón admin **"☁️ Guardar como precargado en Supabase"** en el expander 📂 Fuente de datos. Visible solo para el rol `admin` después de subir un Excel personalizado. Llama a `sio.upload_historico_ventas()`, sube el archivo al path `preloaded/Historico_Ventas.xlsx` del bucket e invalida el caché — el nuevo archivo pasa a ser el precargado permanente para todos los usuarios sin necesidad de acceder al dashboard de Supabase.
- **feat(supabase_io)**: `load_historico_ventas()` — descarga `preloaded/Historico_Ventas.xlsx` desde Storage; devuelve `None` si aún no existe (degradación elegante). `upload_historico_ventas(bytes)` — sube el archivo e invalida el caché. Constante `_EXCEL_CT` para el MIME type de `.xlsx`, compartida por las tres funciones de upload.
- **infra**: Los tres archivos Excel de la carpeta `preloaded/` han sido subidos al bucket `modelos-ml` de Supabase vía API (service key): `veh_ml_features.xlsx` (3.5 MB), `Stock Vehiculos.xlsx` (1.2 MB), `Historico_Ventas.xlsx` (5.0 MB). El deploy en Streamlit Cloud está operativo sin cargar archivos manualmente.

---

### 2026-06-03 (v37)

- **fix(entrenamiento)**: `_load_preloaded()` ya no falla en Streamlit Cloud por archivos Excel ausentes. La función comprueba si los archivos locales (`data/processed/veh_ml_features.xlsx`, `data/raw/Stock Vehiculos.xlsx`) existen antes de leerlos; si no están presentes (entorno de producción, donde `data/` está en `.gitignore`), descarga los datos desde **Supabase Storage** vía `sio.load_datos_precargados()`. El comportamiento en local es idéntico al anterior.
- **feat(supabase_io)**: `load_datos_precargados()` — descarga `preloaded/veh_ml_features.xlsx` y `preloaded/Stock Vehiculos.xlsx` desde el bucket de Supabase y devuelve los dos DataFrames; cacheado 1 hora con `@st.cache_data(ttl=3600)`. `upload_datos_precargados(ventas_bytes, stock_bytes)` — sube ambos archivos al path `preloaded/` en Storage e invalida el caché; destinado a uso admin cuando se actualice el histórico base.
- **infra**: Los archivos de datos precargados deben subirse manualmente una vez al bucket bajo `preloaded/` (vía dashboard Supabase o script local). Ver `docs/05_despliegue.md` para instrucciones.

---

### 2026-06-03 (v36)

- **feat(entrenamiento)**: Datos precargados con caché + opción de Excel personalizado en `pages/1_Entrenamiento.py`. La pestaña **📤 Cargar Datos** ya no bloquea al usuario pidiéndole un Excel: un radio button selecciona entre **📦 Datos precargados** (carga `data/processed/veh_ml_features.xlsx` vía `@st.cache_data` — histórico completo Ene 2017–Mar 2026, ~30 039 registros, incluye `MODELO3`) y **📤 Subir nuevo Excel** (flujo original de carga manual). La lógica de limpieza (`MODELO3` nulos + duplicados por `CHASIS`) se extrajo a `_clean_ventas_df()`, función compartida entre ambas rutas. El stock precargado se lee de `data/raw/Stock Vehiculos.xlsx`. Una vez cargados los datos, se muestra un resumen de registros y columnas sobre el selector.

---

### 2026-06-03 (v35)

- **feat(concesionarios)**: Datos precargados con caché + opción de Excel personalizado en `pages/4_Concesionarios.py`. La página ya no bloquea al usuario pidiéndole un Excel: carga automáticamente `data/raw/Historico_Ventas.xlsx` vía `@st.cache_data` al arrancar. El expander **📂 Fuente de datos** muestra un badge informativo de la fuente activa (precargada o personalizada) y permite subir un Excel propio para sobreescribir los datos durante la sesión. Un botón **↩ Usar precargados** restaura los datos originales sin recargar la página. La lógica de normalización de columnas (`FECHA_VENTA`, `MODELO_NORM`) se extrajo a `_procesar_excel()` para reutilizarla tanto en el loader precargado como en el uploader.

---

### 2026-05-31 (v34)

- **feat(supabase_io)**: `ventas_reales` migrado de Storage JSON a tabla PostgreSQL. La lógica anterior (leer/modificar/escribir un JSON en Storage) tenía riesgo de race condition con escrituras concurrentes. Ahora se usa upsert con `on_conflict="fecha"` sobre la nueva tabla `ventas_reales` en PostgreSQL. Funciones afectadas: `get_ventas_reales()` (ahora lee de DB con `@st.cache_data(ttl=120)`), `save_venta_real()`, `delete_venta_real()`. La función privada `_get_ventas_reales_raw()` fue eliminada.
- **feat(supabase_io)**: Persistencia de resultados Comparativa ML entre sesiones. `save_cml_resultados(run_name, df, ganador)` guarda las métricas como `{run_name}/cml_resultados.json` en Storage al completar la comparativa. `load_cml_resultados(run_name)` las carga con `@st.cache_data(ttl=600)`. El artefacto `cml_resultados.json` añadido a `_ARTIFACT_NAMES`.
- **fix(dashboard)**: Sub-tab **🏆 vs Descartados** ya no requiere que el usuario haya ejecutado la Comparativa en la misma sesión. Si `st.session_state["cml_resultados"]` está vacío, carga automáticamente el JSON persistido en Supabase para el run seleccionado.
- **fix(comparativa)**: Al completar la comparativa con un run de Supabase, guarda los resultados en Storage mediante `sio.save_cml_resultados(run_sel, df_met, mejor)`.
- **db**: Tabla `ventas_reales` creada en Supabase PostgreSQL — `id bigserial PK`, `fecha text NOT NULL UNIQUE`, `ventas integer NOT NULL`, `usuario text`, `timestamp timestamptz DEFAULT now()`.

---

### 2026-05-31 (v33)

- **refactor(auth)**: `guard_page()` añadida a `core/auth_system.py`. Reemplaza el bloque de 5 líneas (`init_session_state` + `check_session_timeout` + `show_login_page` + `st.stop()`) que se duplicaba al principio de cada página. Acepta `app_title`, `permission` opcional y `roles` opcional. Adoptada por `app_principal.py` y todas las páginas de `pages/`.
- **feat(dashboard)**: **Alerta predictiva proactiva** añadida al inicio del Dashboard (antes de los tabs). Si el primer mes del forecast desvía ≥ 15% de la media de los últimos 12 meses, aparece un banner amarillo (demanda alta, `+X%`) o azul (demanda baja, `-X%`) con las unidades previstas, IC 95% y una recomendación de acción operativa. Es forward-looking: no requiere ventas reales registradas.
- **feat(dashboard)**: **Indicador de frescura del modelo** en la barra lateral. Calcula la edad en días del run activo mediante `get_model_age_days()`. Badge verde (< 30 días), amarillo (30–89 días) o rojo (≥ 90 días) con el texto "Reciente / Envejeciendo / Desactualizado" y un aviso `st.sidebar.warning` para modelos con más de 90 días.
- **feat(dashboard)**: **Seguimiento en producción — Real vs Predicción** en el tab 📊 Dashboard. Cuando hay ventas reales registradas (tabla `ventas_reales` en Supabase), muestra KPIs de MAPE de producción, gráfico real vs predicción con IC 95%, tabla detallada mes a mes y tendencia acumulada del error.
- **feat(pages)**: **`pages/7_Registrar_Ventas.py`** — nueva página de feedback loop. Acceso: `admin` y `analyst`. Permite introducir las ventas reales de cada mes una vez cerrado el período mediante formulario (selector mes/año + input unidades). Muestra scoreboard acumulado: MAPE de producción, gráfico real vs predicción con IC 95%, drift alert si algún mes supera el 15% de error, y tabla detallada con gradiente de error. Exportar CSV para roles con permiso `exportar`.
- **feat(pages)**: **`pages/8_Administracion.py`** — nuevo panel de administración. Acceso: solo `admin`. Tres tabs: **👥 Usuarios** (lista de cuentas configuradas en `secrets.toml` con roles e iconos, sin contraseñas), **📜 Audit Log** (tabla de acciones recientes con filtro por acción, KPIs de acciones hoy/total usuarios activos/alertas), **🤖 Gestión de modelos** (tabla de todos los runs con métricas, botón "Aprobar" para marcar un run como activo y botón "Eliminar" con confirmación).
- **feat(supabase_io)**: `get_model_age_days(run_name: str) -> int | None` — calcula los días desde el timestamp del run. `get_ventas_reales() -> list[dict]` y `_get_ventas_reales_raw()` — lectura con caché de la tabla `ventas_reales`. `save_venta_real(fecha, ventas, usuario)` — upsert en Supabase + invalidación de caché. `delete_venta_real(fecha, usuario)` — borrado con audit log. `build_export_excel(...)` y `build_proyeccion_excel(...)` — helpers para generar bytes de Excel en memoria.
- **feat(entrenamiento)**: Expander **"📅 ¿Con qué frecuencia reentrenar? — Evidencia walk-forward"** añadido en el tab Entrenamiento, después de la validación walk-forward. Muestra un gráfico de barras con el MAPE mensual walk-forward coloreado por umbrales (verde < 10%, lima 10–15%, rojo > 15%), líneas guía al 10% y 15%, y un párrafo explicativo de cómo leer la cadencia óptima de reentrenamiento a partir del gráfico.
- **style(styles)**: CSS `context-panel` y clases asociadas (`context-panel-label`, `context-panel-body`, `context-title`, `context-text`, `contexto-kpis`, `ctx-kpi-value`, `ctx-kpi-sub`) añadidas al módulo `styles.py`. Color `primary` actualizado a `#0073FF` (azul vibrante) para acciones de UI principales.
- **refactor(app_principal)**: `@dataclass FeatureCard` para construir las feature cards de forma data-driven. `_get_model_info()` devuelve una tupla `(mape, sarima_order_str)` para mostrar el orden del modelo activo como KPI en la card de Entrenamiento. Logger inicializado con `get_logger("home")`.
- **docs**: README, CHANGELOG y `docs/03_guia_usuario.md` y `docs/02_arquitectura.md` actualizados para reflejar las dos nuevas páginas, el nuevo permiso implícito de administración, el flujo de feedback loop y la tabla de roles revisada.

---

### 2026-05-30 (v32)

- **ux(entrenamiento)**: Guía de pasos siguientes añadida en `pages/1_Entrenamiento.py`. Tab 1 muestra un banner azul persistente cuando los datos ya están cargados: *"Paso siguiente → Ve a la pestaña ✅ Validación"*. Tab 2 muestra un banner verde tras la validación exitosa: *"Paso siguiente → Ve a la pestaña 🤖 Entrenamiento"* (o rojo si hay errores). Elimina la ambigüedad sobre qué hacer después de procesar el Excel.
- **config(entrenamiento)**: Ventana de entrenamiento por defecto actualizada al rango completo de datos disponibles: `TRAINING_DEFAULT_START = 2022-01-01` (antes `2024-01-01`) y nueva constante `TRAINING_DEFAULT_END = 2026-03-31`. Ambas fechas se aplican como valores por defecto en los `date_input` de las pestañas Preparar Datos y Entrenamiento. Maximiza los ciclos estacionales visibles por SARIMA (51 meses / ~4 ciclos) sin sacrificar la ventana completa del piloto.
- **docs**: README y CHANGELOG actualizados — tabla de parámetros configurables refleja las nuevas fechas por defecto.

---

### 2026-05-30 (v31) — Release Final ISDI

- **feat(home)**: Sección **"¿Por qué Chery Tiggo 2?"** añadida en `app_principal.py` antes de las feature cards. Presenta el contexto de selección del piloto: 2,047 unidades históricas, 51+ meses de datos, #1 modelo Chery en volumen, narrativa piloto → escalable. Responde al feedback del jurado sobre justificar la selección del modelo de predicción ante el cliente.
- **feat(dashboard/metricas)**: Sub-tab **"🏆 vs Descartados"** añadido como cuarto sub-tab dentro de 📋 Métricas Técnicas. Si se viene de la página Comparativa ML (`st.session_state["cml_resultados"]` poblado), muestra las métricas en tiempo real de todos los modelos con gráfico de barras y línea objetivo MAPE 15%. Siempre muestra la tabla metodológica de los 5 modelos con justificación de descarte: SARIMA ✅ (estructura AR+MA estacional + exógena, menor MAPE), Prophet ❌ (sobreajuste sin exógena en series cortas), LR ❌ (no captura no-linealidad estacional), RF ❌ (overfitting con n=51), XGB ❌ (no extrapola tendencia).
- **feat(dashboard/recomendaciones)**: Análisis del ciclo estacional añadido al tab Recomendaciones, antes del marco teórico. KPIs de mes pico, mes valle y ratio pico/valle calculados desde los datos históricos reales. Gráfico de barras con media mensual histórica (rojo=máximo, azul=sobre media, gris=bajo media). Callout de negocio sobre el efecto rappel del proveedor en diciembre y la oportunidad de des-estacionalización.
- **feat(proyeccion)**: Calculadora de ROI estratégico **"Valor Estratégico del Sistema — ¿Cuánto vale predecir bien?"** añadida al final de `pages/5_Proyeccion_Ingresos.py`. Inputs configurables: sobrestock actual (uds/mes), costo de financiamiento (%), stockouts mensuales, % de reducción sobrestock/stockout, costo anual del sistema. Outputs: waterfall chart (ahorro sobrestock → ingresos recuperados → valor bruto → costo sistema → valor neto), 4 KPIs (ahorro por sobrestock, ingresos recuperados, valor neto anual, ROI multiplier), tablas comparativas "✅ Con sistema" vs "❌ Sin sistema".
- **feat(escalabilidad)**: Tab **"🚀 Visión del Producto"** añadido como 6.º tab en `pages/6_Escalabilidad.py`. Tres tarjetas de etapa: Etapa 1 HOY (Sistema Reactivo, EN PRODUCCIÓN), Etapa 2 AÑO 1 (Sistema Proactivo — auto-retraining, multi-brand, integración ERP), Etapa 3 AÑO 2+ (Sistema Autónomo — SaaS multi-tenant, optimización de precio, API LatAm). Gráfico dual-axis valor de negocio (barras) + autonomía (línea). Callout de cierre que conecta el piloto Tiggo 2 con la visión LatAm.

---

### 2026-05-30 (v30)

- **perf(supabase_io)**: `@st.cache_data` añadido a 5 funciones que ejecutaban llamadas de red a Supabase en cada recarga de página sin ningún caché: `get_available_runs()` (TTL 5 min), `get_default_run()` (TTL 5 min), `get_runs_df()` (TTL 5 min), `load_acf_pacf_images()` (TTL 10 min), `load_current_model()` (TTL 5 min). `load_precargados()` ya tenía `@st.cache_data(ttl=600)` desde v22.
- **perf(supabase_io)**: Invalidación activa del caché — `st.cache_data.clear()` añadido al final de `approve_model()` y `save_to_dashboard()`. Al entrenar o aprobar un modelo, todas las páginas sirven datos frescos inmediatamente sin esperar la expiración del TTL.

---

### 2026-05-28 (v29)

- **refactor(dashboard)**: Tabs **🔬 ACF/PACF** y **🔍 Grid Search** eliminados de la barra principal de admin/analyst. Ambas vistas se integran como sub-pestañas dentro de **📋 Métricas Técnicas** (sub-tabs: `📊 Resumen · 🔬 ACF/PACF · 🔍 Grid Search`). La barra de tabs pasa de 8 → **6 pestañas** para admin/analyst. Nuevos índices: Walk-Forward → `tabs[3]`, Métricas Técnicas → `tabs[4]`, Asistente IA → `tabs[5]`.
- **docs**: README, CHANGELOG y `docs/03_guia_usuario.md` actualizados para reflejar la nueva estructura de tabs.

---

### 2026-05-28 (v28)

- **refactor(pages)**: Reordenación del menú de navegación — nuevo orden: **1 Entrenamiento → 2 Comparativa ML → 3 Dashboard → 4 Concesionarios → 5 Ingresos**. Archivos renombrados: `4_Comparativa_ML.py` → `2_Comparativa_ML.py`, `2_Dashboard.py` → `3_Dashboard.py`, `5_Concesionarios.py` → `4_Concesionarios.py`, `3_Proyeccion_Ingresos.py` → `5_Proyeccion_Ingresos.py`.
- **config(ingresos)**: Precio medio por unidad actualizado a **15 000 USD** (antes 27 000 USD) en `pages/5_Proyeccion_Ingresos.py`.
- **docs**: README, CHANGELOG y `docs/03_guia_usuario.md` actualizados — árbol de arquitectura, números de app, rutas de archivo y precio por defecto reflejan el nuevo orden.

---

### 2026-05-28 (v27)

- **feat(pages)**: **`pages/5_Concesionarios.py`** — nueva página independiente de análisis y predicciones por concesionario. Accesible para los roles `admin`, `analyst` y `manager`.
- **feat(concesionarios)**: 4 tabs — **📊 Resumen** (barras horizontales de ventas totales + mix de modelos por concesionario), **📈 Evolución Mensual** (líneas por concesionario, share % 100% stacked area, variación MoM), **🔮 Predicciones por Tienda** (shares de los últimos 12 meses aplicados sobre la predicción SARIMA total + IC 95% desagregados por tienda), **📋 Detalle** (ranking + pivot mensual exportable).
- **feat(concesionarios)**: Editor inline de shares (`st.data_editor`) para simular cambios en la distribución (apertura/cierre de tiendas, campañas locales). Los shares editados se renormalizan y se aplican inmediatamente a la predicción.
- **feat(concesionarios)**: KPIs del próximo mes por concesionario — una tarjeta por tienda con predicción central e IC 95%.
- **feat(concesionarios)**: Gráfico histórico + predicción por concesionario con banda IC 95% semitransparente por tienda y línea vertical de corte histó­rico/predicción.
- **feat(concesionarios)**: Barras apiladas de horizonte completo con hover IC 95%, y tabla de predicciones con gradiente de color; exportar CSV para roles con permiso `exportar`.
- **refactor(dashboard)**: Tab **🏪 Concesionarios** eliminado de `2_Dashboard.py` — la funcionalidad se traslada completa a la nueva página propia. Dashboard queda en **8 tabs** para admin/analyst (elimina Concesionarios), **4 tabs** para manager, **2 tabs** para viewer/financiero.

---

### 2026-05-28 (v26)

- **fix(supabase_io)**: `load_current_model()` ahora detecta correctamente el modelo de producción cuando `latest.txt` no existe o está desactualizado. La función ahora sigue el mismo orden de prioridad que `get_default_run()`: primero consulta `activo=TRUE` en PostgreSQL, y solo si falla cae a `latest.txt` como backup. Antes solo leía `latest.txt`, lo que hacía que el tab **Comparación: Nuevo vs Actual** mostrara "No hay modelo previo" aunque hubiera un run activo en DB.

---

### 2026-05-28 (v25)

- **feat(comparativa)**: Nueva sección **2. Período de análisis** en `4_Comparativa_ML.py`. Dos selectboxes "Desde / Hasta" con etiquetas mes/año permiten filtrar el histórico cargado antes de comparar modelos — ya no es obligatorio usar el histórico completo (ej. 111 meses). El filtro aplica al split train/test y a todos los modelos antes de que el usuario configure la partición.
- **ux(comparativa)**: Tooltips (`help=`) añadidos a los selectboxes "Columna de fecha" y "Columna de ventas" en el flujo de carga manual de Excel. Explican qué tipo de columna elegir y muestran ejemplos de nombres habituales (`'Fecha'`, `'Mes'`, `'Ventas'`, `'Unidades'`).
- **refactor(comparativa)**: Renumeración de secciones: Configuración 2→3, Ejecutar comparación 3→4, Resultados 4→5, Publicar 5→6.
- **docs**: README, CHANGELOG y `docs/04_modelos_ml.md` actualizados. README añade sección **Documentación** con enlaces directos a todos los docs desde la página principal de GitHub.

---

### 2026-05-28 (v24)

- **fix(data)**: `data/processed/veh_ml_features.xlsx` ahora incluye la columna **`ventas_otros`** pre-computada. Para cada fila de transacción, `ventas_otros` contiene el conteo mensual de ventas de **otros modelos de la misma marca** vendidos ese mes (total marca – modelo propio). Antes, la columna no existía en el archivo, lo que impedía seleccionarla como variable exógena al subir el archivo manualmente en la página **Comparativa ML**. Ahora está disponible directamente como columna seleccionable (`0–202 uds/mes`, 0 nulos).
- **docs**: README, CHANGELOG y `docs/04_modelos_ml.md` actualizados para documentar la columna `ventas_otros` en `veh_ml_features.xlsx` y su uso en el flujo de subida manual de Comparativa ML.

---

### 2026-05-28 (v23)

- **config(auth)**: `SESSION_TIMEOUT` aumentado de 30 → **60 minutos** en `core/auth_system.py`. Reduce interrupciones de sesión para flujos de trabajo de entrenamiento largos.
- **feat(dashboard)**: Tab **💼 Recomendaciones de Compra** ahora accesible también para el rol `admin` (antes exclusivo de `manager`). El rol admin dispone de 9 tabs en Dashboard (antes 8); índices de ACF/PACF, Grid Search, Walk-Forward, Métricas Técnicas, Asistente IA y Concesionarios desplazados +1.
- **feat(recomendaciones)**: Nuevo expander **📚 Marco teórico — ¿Por qué estas estrategias?** dentro del tab Recomendaciones. Explica el fundamento académico de cada estrategia de compra:
  - **Newsvendor Problem** (Scarf, 1958): justifica usar el percentil 97.5 (IC superior 95%) como base del pedido cuando `c_u > c_o`.
  - **Estrategia Conservadora (+10%)**: stock de seguridad `SS = z · σ_L` (Silver, Pyke & Thomas, 1998) para absorber el error residual del modelo y variabilidad exógena no capturada.
  - **Estrategia Agresiva (+20%)**: nivel de servicio Tipo I ~99%, apropiado cuando el lead time de importación es largo (60–90 días) o la tendencia reciente es creciente.
  - **Señal de tendencia**: fórmula de variación relativa 3 meses vs histórico y sus umbrales de decisión (±10%).
  - **Tabla de limitaciones**: condiciones de validez de los supuestos SARIMA (gaussianidad, estacionariedad, MAPE < 15%).
- **docs**: README, CHANGELOG, `docs/02_arquitectura.md`, `docs/03_guia_usuario.md` y `docs/05_despliegue.md` actualizados para reflejar el nuevo timeout, el acceso extendido al tab Recomendaciones y el marco teórico.

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
