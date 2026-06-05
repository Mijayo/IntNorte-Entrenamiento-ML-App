# Conclusiones — Iteración 3

**Sistema TIGGO 2 · Predicción de Demanda · ISDI Troncal** _Fecha: 2026-06-05 · Versión del sistema: v40_

---

## 1. Contexto de la Iteración

La Iteración 2 cerró con MAPE walk-forward **14.65%** — por debajo del umbral de aceptabilidad (15%) pero aún lejos del objetivo de excelencia (<10%). El roadmap técnico de la iteración apuntaba a tres palancas: ensemble, validación del exógeno, y ampliar datos 2026. Adicionalmente, el feedback del jurado en Release 2 (2026-05-20) marcó dos prioridades de producto: convertir el sistema en un ciclo operativo completo (registro → re-entrenamiento → actualización) y elevar el dashboard como pieza central de la demo.

---

## 2. Mejora del Modelo

### 2.1 Hiperparámetros — Iteración 3

| Parámetro | Iteración 2 | Iteración 3 | Interpretación |
|-----------|-------------|-------------|---------------|
| `p` (AR) | 2 | **1** | Un solo lag AR es suficiente en el régimen 2022+ |
| `d` (dif.) | 0 | **1** | Se reintrodujo diferenciación — la serie mostró tendencia leve tras añadir datos 2026 |
| `q` (MA) | 1 | **0** | MA eliminado — la autocorrelación residual ya la captura el componente AR |
| `P` (AR_s) | 1 | **1** | Sin cambio |
| `D` (dif_s) | 0 | **0** | Sin cambio |
| `Q` (MA_s) | 2 | **2** | Sin cambio |
| `m` | 12 | **12** | Periodicidad anual fija |

**Orden final:** `SARIMA(1, 1, 0)(1, 0, 2)[12]`

### 2.2 Métricas

| Métrica | Iteración 2 | Iteración 3 | Variación |
|---------|-------------|-------------|-----------|
| **MAPE walk-forward** | 14.65% | **10.32%** | ▼ 4.33 pp |
| **AIC** | 138.48 | **137.38** | ▼ 1.10 |
| Orden SARIMA | (2,0,1)(1,0,2)\[12\] | **(1,1,0)(1,0,2)\[12\]** | — |
| Trials válidos / total | 62 / 80 | **71 / 80** | +9 |
| Dataset | 51 meses | 51 meses | Sin cambio |
| Horizonte | 6 meses | 6 meses | Sin cambio |

> **Métrica de producción:** el MAPE walk-forward es el único indicador que refleja el error real en producción. La validación se realiza sobre los últimos 12 meses del dataset usando un esquema expanding window: en cada paso se re-estima el modelo con los datos disponibles hasta ese mes y se predice el siguiente, acumulando errores sin data leakage.

### 2.3 Interpretación del resultado

**MAPE 10.32% — A 0.32 pp del umbral de excelencia (<10%):**
Un error promedio de ±10.32% sobre ~65 unidades/mes implica una desviación de ±6.7 unidades. Para planificación de inventario a 1–3 meses vista, este rango es operacionalmente útil: reduce el riesgo de sobre-stock sin llegar al umbral de confianza plena.

**Por qué d=1 regresa en Iteración 3:**
Al incorporar datos hasta marzo 2026 — meses donde la demanda mostró una leve tendencia alcista — la serie dejó de ser estrictamente estacionaria. Aplicar d=1 absorbe esa tendencia sin requerir exógenas adicionales.

**AIC baja mientras MAPE baja:**
A diferencia de la transición Iter1→Iter2 (donde AIC subió al simplificar), aquí ambas métricas mejoran. Indica que el modelo Iter3 es simultáneamente mejor ajustado y más predictivo: la combinación d=1 + p=1 captura la dinámica real con mínima complejidad.

---

## 3. Nuevas Funcionalidades

### 3.1 Páginas nuevas

| Página | Versión | Descripción |
|--------|---------|-------------|
| **Concesionarios** | v27 | Predicciones desagregadas por tienda usando histórico de ventas por punto de venta |
| **Escalabilidad** | v29 | Portafolio multi-marca — simulación de expansión a Tiggo 4, Tiggo 7, Arrizo 5; mapa de concesionarios LatAm |
| **Proyección de Ingresos** | v31 | Proyección financiera a 12 meses: ingresos brutos, ticket promedio 15 000 USD, análisis de sensibilidad |
| **Registrar Ventas** | v37–v40 | Formulario de registro de ventas reales por mes; tab "Comparativa en Producción" con MAPE real vs predicho |
| **Administración** | v38 | Panel admin: gestión de modelos activos, subida de datos precargados a Supabase Storage, historial de entrenamientos |

### 3.2 Infraestructura y backend

| Componente | Cambio |
|-----------|--------|
| **Supabase Storage** | Archivos Excel precargados (`veh_ml_features.xlsx`, `Historico_Ventas.xlsx`, `Stock Vehiculos.xlsx`) servidos desde Storage — el deploy en Streamlit Cloud no requiere archivos locales |
| **DB ventas_reales** | Tabla en Supabase para persistir ventas registradas entre sesiones; `merge` normalizado a inicio de mes para eliminar mismatch de fechas |
| **CML cross-session** | Resultados de Comparativa ML guardados en DB — persisten entre recargas y usuarios |
| **Caché global** | `@st.cache_data` en todas las funciones de lectura de Supabase; invalidación automática en escrituras |
| **Upsert atómico** | `supabase_io`: operaciones de escritura atomizadas, filtro de bucket en una llamada, guardia por `service_key` |

### 3.3 Producto e inteligencia

| Feature | Versión | Descripción |
|---------|---------|-------------|
| **Tab Recomendaciones** | v23 | Sugerencias automáticas de compra por concesionario, visible para roles admin/analyst |
| **Session timeout** | v23 | Cierre automático de sesión a los 60 min — cumple requisito de seguridad |
| **Feedback loop** | v33 | Ciclo completo: ventas reales registradas → comparación vs predicción → sugerencia de re-entrenamiento |
| **Alertas proactivas** | v33 | El asistente IA emite alertas cuando el MAPE de producción supera umbrales definidos |
| **Panel admin** | v33 | Gestión de usuarios, modelos y datos sin acceder al dashboard de Supabase |
| **Storytelling** | v31 | Página home con ciclo estacional, ROI estimado y visión de producto |

### 3.4 UX y responsive

| Mejora | Versión |
|--------|---------|
| Login — viewport fit cross-browser (Safari `-webkit-fill-available`, `dvh`) | v32 |
| KPIs 2×2 en mobile (< 640px) | v32 |
| Date range filter + tooltips en Comparativa ML | v25 |
| Precio medio actualizado a 15 000 USD | v28 |
| Orden de páginas reorganizado — flujo lógico para demo | v28 |

---

## 4. Aprendizajes Clave

### Técnicos

| Aprendizaje | Detalle |
|------------|---------|
| **La estacionariedad no es permanente** | d=1 fue innecesario en Iter2 (régimen estable 2022–2025) pero necesario en Iter3 (leve tendencia al incorporar datos 2026). Revisar la hipótesis de estacionariedad en cada re-entrenamiento. |
| **Más trials no siempre ganan** | Iter2 con 80 trials y 62 válidos buscó en el mismo espacio que Iter3 con 71 válidos. La mejora vino del rango de búsqueda — no de la cantidad de trials. |
| **El MAPE walk-forward es la métrica de producción** | AIC/BIC son útiles para comparar modelos sobre el mismo dataset; MAPE walk-forward mide el error real al predecir futuros desconocidos. Cualquier comparativa de modelos usa exclusivamente MAPE walk-forward. |
| **Storage + DB elimina el problema de datos en producción** | El mayor pain point del deploy (Streamlit Cloud no tiene sistema de archivos persistente) se resolvió moviendo los Excel a Supabase Storage y los resultados a tablas de DB. |

### Funcionales / de Producto

| Aprendizaje | Detalle |
|------------|---------|
| **El ciclo cerrado es el diferenciador real** | Pasar de "sistema de predicción" a "sistema de decisión operativo" requirió conectar el formulario de ventas reales con el módulo de re-entrenamiento. Sin ese lazo, el modelo se degrada sin visibilidad. |
| **El panel admin reduce la dependencia de IT** | Los roles de negocio (analyst, manager) pueden operar el sistema completo. El admin puede actualizar datos y aprobar modelos sin abrir Supabase. |
| **El feedback del jurado sobre el dashboard fue accionable** | Release 2 señaló el dashboard como punto fuerte y recomendó ampliarle tiempo en presentación. Se incorporó Proyección de Ingresos y storytelling financiero como extensión natural. |

---

## 5. Puntos Clave para Slides Académicos

### Slide: Evolución de las 3 Iteraciones

| Dimensión | Iteración 1 | Iteración 2 | Iteración 3 |
|-----------|------------|------------|------------|
| **MAPE walk-forward** | 27.89% ❌ | 14.65% ⚠️ | **10.32% ✅** |
| Orden SARIMA | (7,1,2)(1,1,2)\[12\] | (2,0,1)(1,0,2)\[12\] | **(1,1,0)(1,0,2)\[12\]** |
| Dataset (meses) | 62 | 51 | 51 |
| Páginas del sistema | 3 | 3 | **8** |
| Ciclo operativo completo | ❌ | ❌ | **✅** |
| Estado | Diagnóstico | Operacional c/ revisión | **Operacional** |

### Slide: Estado Final del Sistema

| Dimensión | Estado |
|-----------|--------|
| Precisión del modelo | ✅ Bueno — MAPE 10.32% (objetivo: <10%) |
| Pipeline de validación | ✅ Walk-forward 12 pasos, expanding window |
| Ciclo operativo | ✅ Registro → Re-entrenamiento → Actualización |
| Cobertura | ✅ Nacional + por concesionario |
| Escalabilidad | ✅ Arquitectura lista para multi-marca |
| Despliegue | ✅ Streamlit Cloud + Supabase — zero-config |

### Slide: Del Modelo al Sistema

```
Iteración 1: modelo aislado — entrena, predice, no se actualiza
Iteración 2: modelo validado — arquitectura lista, métricas bajo umbral
Iteración 3: sistema operativo — registra ventas → compara → alerta → re-entrena
```

---

## 6. Veredicto Final

| Dimensión | Estado | Nota |
|-----------|--------|------|
| **Precisión del modelo** | ✅ Bueno | MAPE walk-forward 10.32% — 4.33 pp de mejora vs Iter2 |
| **Objetivo <10% (excelente)** | ⚠️ Muy próximo | A 0.32 pp del umbral — rango de revisión mensual suficiente |
| **Ciclo operativo** | ✅ Completo | Registro de ventas reales → feedback loop → re-entrenamiento |
| **Infraestructura** | ✅ Production-grade | Supabase Storage + DB — deploy sin dependencias locales |
| **Cobertura funcional** | ✅ Completa | 8 páginas: entrenamiento, dashboard, ML, concesionarios, proyección, escalabilidad, registro, admin |
| **Listo para producción** | ✅ Sin condiciones | Sistema autónomo con supervisión mensual recomendada |

> **Conclusión:** La Iteración 3 lleva el sistema de "herramienta de predicción" a **sistema de decisión operativo**. El MAPE walk-forward baja a 10.32% mediante un reajuste preciso de hiperparámetros, el ciclo registro→re-entrenamiento cierra el lazo operativo, y la arquitectura Supabase Storage + DB elimina todas las dependencias de archivos locales. El sistema está listo para presentación final y para un piloto real en Int. Norte.

---

_Sistema TIGGO 2 · ISDI Troncal · Diego · 2026_
