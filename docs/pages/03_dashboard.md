# Página 3 — Dashboard de Negocio

**Archivo:** `pages/3_Dashboard.py`  
**Acceso:** todos los roles autenticados  
**Icono:** 🚗

---

## Propósito

Página central de análisis y toma de decisiones. Carga automáticamente el modelo SARIMA activo y muestra KPIs, histórico, predicciones, validación walk-forward, métricas técnicas y acceso al asistente IA. El conjunto de tabs disponibles varía según el rol.

---

## Tabs disponibles según rol

| Tab | admin / analyst | manager | viewer |
|-----|:--------------:|:-------:|:------:|
| 📊 Dashboard | ✅ | ✅ | ✅ |
| 🔮 Predicciones | ✅ | ✅ | ✅ |
| 💼 Recomendaciones | ✅ | ✅ | — |
| 🔄 Walk-Forward | ✅ | — | — |
| 📋 Métricas Técnicas | ✅ | — | — |
| 🤖 Asistente IA | ✅ | ✅ | — |

---

## Sidebar

- **Selector de versión del modelo** — `selectbox` con todos los runs disponibles. El run activo se marca como "🟢 Activo en producción"; los históricos como "🔵 Versión histórica".
- **Indicador de frescura del modelo** — calculado como días desde el entrenamiento:
  - 🟢 < 30 días: Reciente
  - 🟡 30–89 días: Envejeciendo
  - 🔴 ≥ 90 días: Desactualizado (añade warning de reentrenamiento)

---

## Alerta predictiva proactiva

Se activa automáticamente cuando la predicción del próximo mes desvía ≥ 15% de la media de los últimos 12 meses:

- **Demanda inusualmente alta** → banner amarillo con recomendación de anticipar el pedido.
- **Demanda inusualmente baja** → banner azul con recomendación de reducir el pedido.

---

## Realtime watcher (`@st.fragment(run_every=30)`)

Cada 30 segundos comprueba si hay un nuevo run en Supabase. Si aparece uno, muestra un toast de notificación y hace `st.rerun()`.

---

## Tab 📊 Dashboard

**KPIs principales:**

| KPI | Fuente |
|-----|--------|
| Total Ventas | `metricas['datos_limpios']['total_ventas']` |
| Meses de Datos | `metricas['datos_limpios']['meses_datos']` |
| MAPE | `metricas['walk_forward_validation']['mape']` (rojo > 15%, ámbar > 10%) |
| Próximo Mes | `metricas['predicciones_futuras']['proximo_mes']` uds |

**Gráfico de serie temporal histórica** — línea con área rellena y línea de media histórica (punteada).

**Estadísticas descriptivas** *(admin/analyst únicamente)* — promedio, mediana, mínimo, máximo, desviación estándar + parámetros SARIMA y AIC/BIC.

**Seguimiento en producción** — si hay ventas reales registradas (ver página 7), muestra un panel de drift con gráfico real vs predicción y alerta si el error máximo supera el 15%.

**Contexto de Mercado** *(admin/analyst/manager)* — market share automotriz estimado Perú 2024 (Toyota, Hyundai, Kia, **Chery**, Chevrolet, etc.) con barra destacada para Chery y callout de posición competitiva.

---

## Tab 🔮 Predicciones

**Banner educativo** — explica la diferencia entre:
- ① Predicción mes a mes (independiente por mes, no un agregado dividido).
- ② Horizonte de 6 meses (ventana de visibilidad; la validación walk-forward — zona violeta — simula el proceso real).

**KPIs:**

| KPI | Descripción |
|-----|-------------|
| Próximo Mes | Predicción central del mes inmediato en uds |
| Total Horizonte | Suma de todas las predicciones del horizonte |
| Promedio Mensual | Media de las predicciones del horizonte |
| MAPE real (1 mes) | MAPE walk-forward (objetivo < 15%) |

**Gráfico principal** — combina:
- Histórico (verde claro).
- Región walk-forward sombreada en violeta con etiqueta.
- Predicciones walk-forward mes a mes (diamantes violetas, línea punteada).
- Predicción futura (línea naranja continua con círculos).
- Banda IC 95% (relleno semitransparente rojo).
- Línea vertical punteada que separa histórico de predicción futura.

**Tablas:**
- Predicción futura: Mes · Predicción · IC_Inferior · IC_Superior.
- Walk-forward: Mes · Real · Predicción · Error Abs · Error % (con gradiente de color).

**Exportar Excel** *(roles con permiso `exportar`)* — incluye predicciones + walk-forward + histórico.

**Simulador de Escenarios** *(expander)* — slider de ajuste de demanda de -50% a +100% en pasos de 5%. Muestra gráfico de base vs escenario ajustado con banda IC, y métricas de total base, total escenario y factor aplicado.

---

## Tab 💼 Recomendaciones de Compra

*(admin, analyst, manager)*

**Análisis del próximo mes:**
- Predicción puntual y rango IC 95%.
- **Estrategia Conservadora** — IC superior × 1.10 (minimiza sobrestock).
- **Estrategia Agresiva** — IC superior × 1.20 (maximiza cobertura).
- Señal de tendencia (últimos 3m vs histórico): CRECIENTE (+10%) → estrategia agresiva; DECRECIENTE (−10%) → conservadora; ESTABLE → predicción directa.

**Cadena de Suministro — ¿Cuándo hacer el pedido?** *(nueva sección v41)*

Sección accionable con datos reales Chery 2025:

| Elemento | Descripción |
|----------|-------------|
| KPI Lead time | "22–24 días" (promedio real 2025) |
| KPI Pedido óptimo | Fecha calculada como `Fecha_inicio_mes - 23 días` |
| KPI Días al deadline | Días hasta el deadline óptimo, con semáforo (verde / ámbar / rojo) |
| Timeline visual | 4 hitos: Conservador (–30d), Óptimo (–23d), Agresivo (–15d), Inicio del mes |
| Mapa logístico | Puerto Callao → Almacén Lima → Piura/Chiclayo → Tarapoto/Cajamarca *(opt.)* |
| Nota financiera | 0% interés primeros 60 días en stock → 8% anual a partir del día 61 |

**Análisis del Ciclo de Valor:**
- KPIs: mes pico histórico, mes valle, ratio pico/valle.
- Gráfico de barras de media histórica por mes del año.
- Callout con insights de negocio: efecto rappel del proveedor, oportunidad de des-estacionalización.

**Expander "📚 Marco teórico — ¿Por qué estas estrategias?":**
- Newsvendor Problem (Scarf, 1958).
- Stock de seguridad SS = z · σ_L (Silver, Pyke & Thomas, 1998).
- Nivel de servicio Tipo I (Zipkin, 2000): lead time real Chery **15–30 días** (promedio 2025: 22–24 días).
- Tabla de condiciones de validez y limitaciones.

---

## Tab 🔄 Walk-Forward Validation

*(admin, analyst)*

**KPIs:** MAPE promedio, mejor mes, peor mes, meses evaluados.

**Gráfico** — líneas de real vs predicción walk-forward por mes.

**Tabla** — con gradiente de color en columna Error %.

---

## Tab 📋 Métricas Técnicas

*(admin, analyst)* — cuatro sub-tabs:

### Sub-tab 📊 Resumen

Parámetros SARIMA (order, seasonal_order) y métricas de ajuste (AIC, BIC, MAPE WF, período de datos, horizonte).

### Sub-tab 🔬 ACF/PACF

Gráficos de autocorrelación y autocorrelación parcial guardados durante el entrenamiento.

### Sub-tab 🔍 Grid Search

- KPIs: combinaciones evaluadas, mejor MAPE, AIC seleccionado.
- Top 10 modelos por MAPE con gradiente de color.
- Scatter AIC vs MAPE coloreado por parámetro p.

### Sub-tab 🏆 vs Descartados

Muestra los resultados de la Comparativa ML para el run activo:
- Si se ejecutó la Comparativa ML en la sesión actual o está guardada en Supabase → tabla de métricas de los 5 modelos + gráfico de barras MAPE con línea de objetivo 15%.
- Si no → muestra la justificación metodológica de por qué se eligió SARIMA sobre las otras familias (tabla con estacionalidad, variable exógena, requisito mínimo y motivo de descarte).

---

## Tab 🤖 Asistente IA

*(admin, analyst — rol técnico / manager — rol negocio)*

Consulta en lenguaje natural a **Gemini 2.5 Flash** sobre el modelo activo.

**Contexto enviado automáticamente:**
- Modelo SARIMA y parámetros.
- AIC, BIC, MAPE walk-forward.
- Predicción próximo mes e IC 95%.
- Predicción total del horizonte.
- Tendencia de los últimos 3 meses vs histórico.
- Período de datos y total de ventas.
- **Cadena de suministro Chery** *(v41)*: lead time min/avg/max, ruta logística, sedes, tasa de inventario (60d gratis + 8% anual).

**Caché LLM:** las respuestas se cachean en Supabase por run. Si el mismo run ya tiene respuestas guardadas, se recuperan instantáneamente sin llamar a la API.

**Ejemplos de preguntas (rol técnico):**
- *"¿Por qué el MAPE walk-forward es mayor que el MAPE de Optuna?"*
- *"¿Qué significa P=1 en el componente estacional?"*
- *"¿Es un buen modelo con AIC de 450?"*

**Ejemplos de preguntas (rol gerente):**
- *"¿Debería pedir más o menos unidades que el mes pasado?"*
- *"¿Cuál es el rango realista de ventas para el próximo mes?"*
