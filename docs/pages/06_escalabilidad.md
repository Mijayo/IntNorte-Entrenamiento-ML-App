# Página 6 — Escalabilidad — Plataforma Multi-Marca

**Archivo:** `pages/6_Escalabilidad.py`  
**Acceso:** todos los roles autenticados  
**Icono:** 🌐

---

## Propósito

Página informativa (no requiere ninguna acción del usuario) que demuestra cómo el pipeline SARIMA se generaliza a otras marcas, modelos de vehículo, líneas de negocio y mercados geográficos. Plasma la hoja de ruta técnica y estratégica del sistema.

---

## KPI Banner

| KPI | Valor | Descripción |
|-----|------:|-------------|
| Marcas Compatibles | ∞ | Cualquier marca con historial ≥ 36 meses |
| Semanas para Onboarding | 2–4 | Desde datos raw hasta dashboard en vivo |
| Datos Mínimos | 36 meses | 3 ciclos estacionales completos para SARIMA |
| Líneas de Negocio | 6+ | Vehículos, repuestos, seguros, servicio… |

---

## Tab 🏗️ Arquitectura

### Pipeline genérico Brand-Agnostic

Diagrama visual del pipeline de 6 nodos con código HTML/CSS embebido:

```
📄 Datos de Entrada  →  ✅ Validación  →  🔍 Optuna TPE
→  🤖 SARIMAX  →  ☁️ Supabase  →  📊 Dashboard
```

- Nodos **verde** (`n-active`): específicos por marca (solo el archivo Excel).
- Nodos **azul** (`n-generic`): genéricos y completamente reutilizables.

**Insight clave:** el único componente que cambia entre marcas es el archivo de entrada.

### Lo que cambia vs lo que se reutiliza

| Lo que cambia por marca | Lo que se reutiliza sin cambios |
|------------------------|--------------------------------|
| Archivo Excel | Pipeline completo (validación → Optuna → SARIMA → walk-forward) |
| Filtro MARCA | Variable exógena automática (Pearson r ≥ 0.3) |
| Filtro MODELO3 | Supabase Storage + PostgreSQL (misma instancia) |
| Fecha inicio | Dashboard multi-rol RBAC |
| Máx. ventas/mes | Comparativa 5 modelos ML |
| Horizonte | Proyección de Ingresos (ajustable) |
| — | Asistente IA Gemini (contexto cargado automáticamente) |
| — | Sistema de autenticación (Supabase Auth + roles) |
| — | 17 tests unitarios (cobertura total) |

### El sistema ya es multi-marca hoy

La tabla `training_runs` ya tiene columnas `marca` y `modelo`. Para soporte completo multi-tenant:
1. Añadir índice único `activo_marca_modelo` por par *(marca, modelo)*.
2. Agregar selector Marca/Modelo al sidebar del Dashboard.
3. Namespacing en Storage: `{marca}/{modelo}/runs/{run_name}/`.

**Estimación: 1–2 días de desarrollo.**

---

## Tab 🚗 Portafolio

Estado de expansión a otros modelos del portafolio Chery y otras marcas:

| Estado | Color | Descripción |
|--------|-------|-------------|
| ✅ Activo | Verde | En producción — modelo entrenado y publicado |
| 🔄 En evaluación | Azul | Datos ≥ 36 meses disponibles — listo para entrenar |
| 📋 Pendiente | Amarillo | Historia < 36 meses — requiere espera |
| 🎯 Potencial | Violeta | Marca externa — requiere acceso a datos |

**Modelos del portafolio** — Chery: promedio real últimos 12 meses desde `veh_ml_features.xlsx`; JAC/BYD/MG: estimaciones de mercado.

| Marca | Modelo | Segmento | Dem. Est. (uds/mes) | Historial (m) | Fuente |
|-------|--------|----------|--------------------:|:-------------:|--------|
| CHERY | TIGGO 2 | SUV Compacto | 48 | 110 | Real (12m avg) |
| CHERY | TIGGO 4 PRO | SUV Compacto | 14 | 51 | Real → TIGGO 4 |
| CHERY | ARRIZO 5 | Sedán | 9 | 54 | Real (12m avg) |
| CHERY | TIGGO 5X | SUV Mediano | 5 | 10 | Estimación (sin MODELO3 exacto) |
| CHERY | TIGGO 7 PRO | SUV Mediano | 3 | 76 | Real → TIGGO 7 |
| CHERY | TIGGO 8 PRO | SUV Grande | 2 | 70 | Real → TIGGO 8 |
| JAC | HUNTER PLUS | Pick-up 4x4 | 55 | 0 | Est. mercado |
| JAC | SEI 7 | SUV Grande | 20 | 0 | Est. mercado |
| BYD | ATTO 3 | SUV Eléctrico | 30 | 0 | Est. mercado |
| MG | ZS | SUV Compacto | 35 | 0 | Est. mercado |

> La demanda Chery se calcula con `_cargar_dem_mensual_real()` (`@st.cache_data ttl=3600`): agrupa `veh_ml_features.xlsx` por `(MODELO3, PERIODO)`, filtra los últimos 12 meses y promedia las unidades mensuales. El historial es el número de periodos únicos con al menos 1 venta. Si el Excel no está disponible, se usan los fallbacks hardcodeados.

Gráfico de barras horizontales coloreado por estado + `st.caption()` con fuente de datos + tabla detallada con MAPE estimado.

---

## Tab 💼 Líneas de Negocio

Aplicación del framework de predicción a líneas adyacentes del negocio automotriz:

| Línea de Negocio | Estado | Algoritmo | MAPE Esperado | Sem. Implementar |
|-----------------|--------|-----------|:-------------:|:----------------:|
| 🚗 Vehículos Nuevos | ✅ Activo | SARIMAX + Optuna | 10–15% | 0 |
| 🔧 Repuestos y Accesorios | 🔄 Siguiente paso | SARIMAX / LSTM | 12–18% | 3 |
| 🛡️ Seguros de Vehículo | 📋 Planificado | SARIMA + exog ventas | 10–14% | 2 |
| 🔩 Servicio Post-Venta | 📋 Planificado | SARIMAX + flota exog | 14–20% | 4 |
| 🚙 Vehículos Usados (CPO) | 🎯 Potencial | SARIMA + exog economía | 15–22% | 5 |
| 💳 Financiamiento Vehicular | 🎯 Potencial | Prophet / SARIMA | 12–18% | 4 |

Radar chart con 5 dimensiones por línea: disponibilidad de datos, ROI potencial, facilidad de implementación, madurez de datos, inmediatez.

---

## Tab 📋 Playbook de Onboarding

Guía paso a paso para incorporar una nueva combinación marca/modelo en **2 a 4 semanas**:

**Timeline Gantt** (semanas 1–4):
- **Semana 1** — Extracción + limpieza de datos.
- **Semana 2** — Validación automática + primer entrenamiento + ACF/PACF.
- **Semana 3** — Walk-forward + UAT con stakeholders + ajuste de parámetros financieros.
- **Semana 4** — Aprobación + monitoreo 30 días + documentación del run.

**Requisitos de datos mínimos:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `FECHA-VENTA` | Fecha | Fecha de cada transacción |
| `MARCA` | Texto | Nombre de la marca |
| `MODELO3` | Texto | Nombre del modelo |

Condiciones adicionales: ≥ 36 meses continuos, ≤ 5% de datos faltantes, sin preprocesamiento previo requerido.

**Checklist de go-live** — 13 ítems organizados en categorías: Datos, Modelo, Sistema, Dashboard, Financiero, Concesionarios, Aprobación.

---

## Tab 🌎 Expansión Geográfica

Roadmap de despliegue en mercados LatAm:

| Fase | País | Distribuidor | Chery Share | Prioridad |
|------|------|-------------|:-----------:|:---------:|
| Fase 0 — Actual | 🇵🇪 Perú | Interamericana Norte | ~8% | 1 |
| Fase 1 (6–12m) | 🇨🇴 Colombia | Grupo Automotriz Chery CO | ~5% | 2 |
| Fase 1 (6–12m) | 🇪🇨 Ecuador | Autec S.A. | ~9% | 2 |
| Fase 2 (12–24m) | 🇧🇴 Bolivia | TBD | ~6% | 3 |
| Fase 2 (12–24m) | 🇵🇾 Paraguay | TBD | ~4% | 3 |
| Fase 2 (12–24m) | 🇺🇾 Uruguay | TBD | ~3% | 3 |
| Fase 3 (24+m) | 🇨🇱 Chile | Gildemeister | ~4% | 4 |
| Fase 3 (24+m) | 🇲🇽 México | TBD | ~2% | 4 |

Gráfico Gantt horizontal por país + tabla detallada + tarjetas de factores clave de éxito (Datos, Infraestructura, Organización).

---

## Tab 🚀 Visión del Producto

Tres etapas de evolución del sistema:

| Etapa | Horizonte | Estado | Descripción clave |
|-------|-----------|--------|-------------------|
| 1 — Reactivo | HOY (2025–2026) | ✅ EN PRODUCCIÓN | Predicción mensual SARIMA, dashboard RBAC, asistente Gemini, proyección financiera |
| 2 — Proactivo | Año 1 (2026–2027) | 📋 ROADMAP | Auto-retraining, multi-marca, alertas push, integración ERP |
| 3 — Autónomo | Año 2+ (2027–2028) | 🎯 VISIÓN | SaaS multi-tenant, optimización de precios, API LatAm pública |

Gráfico dual-axis: valor de negocio (barras) y autonomía operativa % (línea punteada).
