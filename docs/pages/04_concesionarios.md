# Página 4 — Concesionarios

**Archivo:** `pages/4_Concesionarios.py`  
**Acceso:** `admin`, `analyst`, `manager`  
**Icono:** 🏪

---

## Propósito

Análisis histórico de ventas por tienda (concesionario) y predicciones desagregadas. El modelo SARIMA predice el total nacional; la página reparte esa predicción entre concesionarios usando los shares históricos de los últimos 12 meses.

---

## Fuente de datos

La página resuelve la fuente en el siguiente orden de prioridad:

1. **Archivo personalizado en sesión** — si el usuario subió un Excel propio, tiene preferencia.
2. **Datos precargados** — `data/processed/veh_ml_features.xlsx` desde Supabase Storage o disco local.

El estado activo se muestra en el expander **📂 Fuente de datos**, que incluye:
- Badge con la fuente activa y el número de registros CHERY.
- Uploader para subir un Excel personalizado (normalización automática de columnas).
- Botón **↩ Usar precargados** (aparece cuando hay un archivo personalizado activo).
- Botón **☁️ Guardar como precargado en Supabase** *(solo admin)* — sube el archivo a Supabase Storage para que sea el precargado permanente para todos los usuarios.

**Columnas de fecha aceptadas:** `FECHA_VENTA`, `FECHA-VENTA`, `FECHA VENTA`.  
**Columnas de modelo aceptadas:** `MODELO2`, `MODELO3`, `MODELO`.  
**Columnas de concesionario detectadas:** `CONCESIONARIO`, `DET_CC`, `AGE`, `SUCURSAL`.

---

## Sidebar — Selector de modelo y filtros

- **📦 Versión del modelo** — selecciona el run SARIMA para las predicciones.
- **Filtros (sidebar):**
  - Año (multiselect)
  - Modelo de vehículo (selectbox — todos o un modelo específico)
  - Concesionarios a incluir (multiselect)

---

## KPIs globales

| KPI | Descripción |
|-----|-------------|
| Total Ventas CHERY | Número de registros tras aplicar filtros |
| Concesionarios | Número de tiendas distintas |
| Top Concesionario | Tienda con mayor volumen |
| Último Dato | Fecha de la venta más reciente |

---

## Tab 📊 Resumen

**Ventas totales por concesionario** — barras horizontales con unidades y % del total.

**Mapa geográfico de Perú** — `px.scatter_mapbox` sobre tiles OpenStreetMap (sin API key).
- Burbujas con tamaño proporcional a ventas y color por concesionario (paleta `COLORS['series']`).
- Tooltip: nombre, unidades vendidas y % del total.
- Zoom inicial `5.5`, centrado en `(-9.19, -75.0)`.
- **Controlador de zoom** — barra de herramientas con tres botones (`zoomInMapbox`, `zoomOutMapbox`, `resetViewMapbox`) en la esquina superior derecha. Configurado vía `modeBarButtonsToKeep`; scroll con ratón y drag siguen disponibles.
- Matching automático: `_coords_concesionario()` busca el nombre de ciudad dentro del nombre del concesionario. Ciudades cubiertas: Lima, Callao, Piura, Chiclayo, Tarapoto, Cajamarca, Trujillo, Arequipa, Cusco, Iquitos, Huancayo, Puno.
- Si ningún concesionario contiene una ciudad reconocible, se muestra un `st.info()` indicando que el nombre debe incluir la ciudad.

**Distribución de modelos por concesionario** — barras apiladas 100% coloreadas por modelo.

---

## Tab 📈 Evolución Mensual

**Líneas de ventas mensuales** por concesionario con `hovermode='x unified'`.

**Share mensual (%)** — área 100% apilada por concesionario.

**Crecimiento MoM (%)** — barras agrupadas de variación mes a mes (`pct_change()`). Solo se muestra si hay ≥ 2 meses de datos.

---

## Tab 🔮 Predicciones por Tienda

### Metodología (banner)

> El modelo SARIMA predice el **total nacional** de ventas TIGGO 2.  
> Para desglosar por concesionario se calcula el **share de los últimos 12 meses** de cada tienda y se aplica como ponderación sobre la predicción total y sus intervalos de confianza.  
> **Supuesto:** la distribución relativa entre concesionarios se mantiene estable en el horizonte de predicción.

### Cálculo de shares

- Se toman los registros de los últimos 12 meses (todos los concesionarios, sin filtro de concesionario del sidebar).
- Se calcula `shares_pct = ventas_conc / ventas_totales` para cada tienda.
- Si el filtro de concesionarios del sidebar está activo, se filtran los shares y se renormalizan a 100%.

### Editor de shares *(expander)*

Permite ajustar el porcentaje de cada tienda para simular cambios estructurales (apertura/cierre, campaña local). Los shares deben sumar 100%; si no, aparece advertencia.

### KPIs del próximo mes

Una tarjeta por concesionario con unidades predichas y rango IC 95%.

El mes mostrado es el **primer mes futuro** con `Fecha >= hoy`. Si todas las predicciones almacenadas ya son pasadas (modelo sin reentrenar), se muestra la más reciente disponible. La columna `Mes` siempre se regenera en español desde `Fecha` (`_MESES_ES`), independientemente del idioma con que se guardó el modelo en Supabase.

### Gráfico principal

Histórico real (líneas sólidas) + predicción (líneas punteadas con diamantes) + banda IC 95% (relleno semitransparente), una traza por concesionario. Línea vertical punteada separa histórico de predicción.

### Barras de horizonte completo

Barras apiladas por mes con predicción de cada concesionario y hover con IC 95%.

### Tabla de predicciones

Columnas: Mes · Concesionario · Share (%) · Predicción · IC_Inferior · IC_Superior. Incluye filas de **TOTAL** por mes. Gradiente de color en columna Predicción.

Exportable como CSV *(roles con permiso `exportar`)*.

---

## Tab 📋 Detalle

**Ranking acumulado** — tabla con Ventas · % Total · Acumulado % (permite análisis de Pareto).

**Pivot mensual** — matriz mes × concesionario con gradiente de color. Exportable como CSV *(roles con permiso `exportar`)*.
