# Página 5 — Proyección de Ingresos

**Archivo:** `pages/5_Proyeccion_Ingresos.py`  
**Acceso:** `admin`, `analyst`, `financiero` (permiso `ver_ingresos`)  
**Icono:** 💰

---

## Propósito

Traduce la predicción SARIMA del modelo activo en cifras financieras en USD. Permite configurar precio unitario, margen neto y tipo de cambio para generar un escenario financiero a 6 meses. Incluye además una calculadora de ROI estratégico del sistema.

> Página extraída del Dashboard como página independiente el 2026-05-27.

---

## Sidebar

- **📦 Versión del modelo** — selector de run. Carga las predicciones del run seleccionado.

---

## Tab 💰 Proyección de Ingresos

### Inputs de escenario

| Campo | Por defecto | Descripción |
|-------|:-----------:|-------------|
| Precio medio por unidad (USD $) | 15 000 | Precio de venta neto por vehículo en USD |
| Margen neto estimado (%) | 8.0 | % de beneficio neto sobre ingresos. 0 = omitir columna de beneficio |
| Tipo de cambio (USD / moneda local) | 1.00 | Multiplica el precio para convertir a moneda local si aplica |

> El precio efectivo = `precio_usd × tc`. Si ya está en USD, deja el tipo de cambio en 1.

### Cálculo

```
Ingresos ($) = Predicción_uds × precio_usd × tc
IC Inf ($)   = IC_Inferior_uds × precio_usd × tc
IC Sup ($)   = IC_Superior_uds × precio_usd × tc
Beneficio ($) = Ingresos ($) × (margen_pct / 100)  [si margen > 0]
```

### KPIs

| KPI | Descripción |
|-----|-------------|
| Unidades (6 meses) | Suma de predicciones del horizonte |
| Ingresos centrales (6 m) | Ingresos totales a precio central |
| Rango IC 95% (6 m) | Rango pesimista–optimista en USD |
| Beneficio neto (6 m) | Solo si margen > 0 |
| Margen aplicado | Solo si margen > 0 |

### Gráfico de ingresos

Barras superpuestas con dos capas:
1. **Rango IC 95%** — barra semitransparente ámbar desde IC_Inf a IC_Sup.
2. **Ingresos proyectados** — barra sólida verde con etiqueta de valor.
3. **Beneficio neto** — línea punteada naranja con diamantes *(si margen > 0)*.

### Tabla mes a mes

Columnas: Mes · Predicción (uds) · Ingresos ($) · IC Inf ($) · IC Sup ($) · Beneficio ($) *(si margen > 0)*.  
Fila de **TOTAL** al final.  
Gradiente de color en columna Ingresos ($).

**Exportar Excel** *(roles con permiso `exportar`)* — genera archivo con todas las columnas.

---

## Tab 💎 Valor Estratégico del Sistema

Calculadora interactiva para cuantificar el retorno económico del sistema de predicción frente a la gestión sin predicción.

### Inputs (tres columnas)

**Parámetros de inventario:**

| Campo | Por defecto | Descripción |
|-------|:-----------:|-------------|
| Sobrestock promedio sin predicción (uds/mes) | 5 | Unidades extra compradas en exceso por mes |
| Costo mensual de capital inmovilizado (%) | 1.5 | % del precio/ud que cuesta tener la unidad en inventario |

**Parámetros de venta perdida:**

| Campo | Por defecto | Descripción |
|-------|:-----------:|-------------|
| Ventas perdidas por mes sin predicción (uds/mes) | 2 | Unidades no vendidas por quiebre de stock |
| Reducción de stockout con el sistema (%) | 70 | % de stockouts evitados con mejor previsión |

**Parámetros del sistema:**

| Campo | Por defecto | Descripción |
|-------|:-----------:|-------------|
| Reducción de sobrestock con el sistema (%) | 60 | % de exceso de inventario evitado |
| Costo anual del sistema (USD $) | 1 200 | Streamlit Cloud + Supabase + mantenimiento (~$100/mes) |

### Cálculo

```
costo_por_ud_mes        = precio_usd × (costo_fin_pct / 100)
ahorro_sobrestock_anual = sobrestock × (reduc_ss / 100) × costo_por_ud_mes × 12
margen_por_ud           = precio_usd × (margen_pct / 100)
ahorro_stockout_anual   = stockout × (reduc_so / 100) × margen_por_ud × 12
valor_bruto_anual       = ahorro_sobrestock_anual + ahorro_stockout_anual
roi_neto                = valor_bruto_anual − costo_sistema_anual
roi_ratio               = valor_bruto_anual / costo_sistema_anual
payback_meses           = costo_sistema_anual / (valor_bruto_anual / 12)
```

### KPIs del ROI

| KPI | Descripción |
|-----|-------------|
| Ahorro sobrestock/año | Capital inmovilizado ahorrado anualmente |
| Ingresos recuperados/año | Margen de ventas recuperadas al evitar stockouts |
| Valor neto anual del sistema | Beneficio neto tras descontar el costo |
| ROI del sistema | Múltiplo: cuántas veces el costo se recupera |

### Gráfico Waterfall

Muestra el flujo completo: Ahorro sobrestock → Ingresos recuperados → Valor bruto → Descuento costo sistema → **Valor neto**.

### Tablas comparativas

**✅ Con sistema de predicción** vs **❌ Sin sistema de predicción** — comparación de sobrestock, stockouts, capital inmovilizado, margen perdido y payback.
