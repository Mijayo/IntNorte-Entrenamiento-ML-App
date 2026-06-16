# Página 7 — Registrar Ventas Reales

**Archivo:** `pages/7_Registrar_Ventas.py`  
**Acceso:** `admin`, `analyst`  
**Icono:** 📋

---

## Propósito

Feedback loop de producción. Permite ingresar las ventas reales de cada mes una vez cerrado el período. El sistema las compara automáticamente con la predicción del modelo activo, calcula el error real en producción y activa alertas de drift si el modelo pierde precisión.

**Los datos registrados se usan para:**
- Comparar forecast vs ventas reales mes a mes.
- Detectar drift automáticamente (error > 15% activa alerta).
- Mostrar el scoreboard acumulado de precisión del modelo en producción.
- Alimentar el panel de "Seguimiento en Producción" del Dashboard.

---

## Sidebar

- **📦 Versión del modelo** — selecciona el run contra el que comparar las ventas reales.

---

## Tab ➕ Registrar mes

### Formulario de registro

| Campo | Descripción |
|-------|-------------|
| Mes | Selectbox con los 12 meses (por defecto: mes anterior al actual) |
| Año | Number input (2020 – año actual) |
| Unidades vendidas (real) | Number input entero ≥ 0 |

Al pulsar **💾 Guardar registro**, el dato se guarda en Supabase con timestamp y usuario.

> Si se introduce 0 unidades, aparece advertencia — registra al menos 1 si las ventas fueron 0 y ajusta después.

### Historial de registros

Tabla ordenada por fecha descendente con: Mes · Unidades reales · Registrado por · Timestamp.

**Eliminar registro** *(solo admin)* — expander con selectbox de mes a eliminar y botón de confirmación. La eliminación es irreversible.

---

## Tab 📡 Comparativa en producción

Disponible una vez que hay al menos un mes registrado que coincida con el horizonte de predicción del run seleccionado.

> **Nota de alineación:** las fechas de ventas reales se normalizan a inicio de mes (`to_period('M').to_timestamp()`) para hacer match con las fechas de predicción del modelo (frecuencia `ME`).

### KPIs de producción

| KPI | Color | Descripción |
|-----|-------|-------------|
| MAPE producción | Rojo > 15%, ámbar > 10% | Error medio entre predicción y real, todos los meses registrados |
| Dentro del IC 95% | — | % de meses cuya venta real cayó dentro del IC 95% del modelo |
| Mejor mes | Verde | Mes con menor error porcentual (formato: "Ene 2026") |
| Peor mes | Rojo | Mes con mayor error porcentual (formato: "Ene 2026") |

> **i18n:** los nombres de mes se muestran en español mediante `_MESES_ES` / `_mes_es()`. Aplica también a los tooltips del gráfico y a la columna Mes de la tabla detallada.

### Alertas de drift

- **Error > 15%** → error rojo recomendando reentrenamiento.
- **Error 10–15%** → advertencia amarilla "aceptable pero cerca del umbral".
- **Error ≤ 10%** → éxito verde.

### Gráfico real vs predicción

Tres capas:
1. **Banda IC 95%** — relleno semitransparente rojo.
2. **Predicción del modelo** — línea punteada naranja con diamantes.
3. **Real registrado** — línea verde sólida con círculos.

### Tabla detallada

Columnas: Mes · Real · Predicción · Error Abs · Error % · Dentro IC 95%.  
Gradiente de color en columna Error % (verde → amarillo → rojo).

---

## Flujo operativo mensual recomendado

1. Al cerrar el mes (p.ej. el día 5 del mes siguiente), entra a esta página.
2. Selecciona el mes y año recién cerrado.
3. Introduce las unidades vendidas reales (dato del ERP/DMS del distribuidor).
4. Haz clic en **💾 Guardar registro**.
5. Ve a la pestaña **📡 Comparativa en producción** para ver el error real.
6. Si el MAPE supera el 15%, ve a **Entrenamiento** y lanza un nuevo modelo.
