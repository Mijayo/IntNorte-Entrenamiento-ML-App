# Conclusiones — Iteración 1
**Sistema TIGGO 2 · Predicción de Demanda · ISDI Troncal**
*Fecha: 2026-04-22 · Versión del sistema: v15*

---

## 1. Contexto del Proyecto

**Objetivo:** Construir un sistema de predicción de demanda mensual para el Chery Tiggo 2 en Perú, con interfaz web operacional para usuarios de negocio (gerencia, analistas, concesionarios).

**Stack principal:**
- Frontend/App: Streamlit (Python)
- Modelos ML: statsmodels (SARIMA), Prophet, scikit-learn, XGBoost
- Optimización: Optuna (TPE — Tree-structured Parzen Estimator)
- Backend: Supabase (Auth + PostgreSQL + Storage)
- Despliegue: Streamlit Cloud

**Dataset:** 62 meses de ventas (enero 2021 – febrero 2026) · 2.047 unidades totales

---

## 2. Modelos ML Utilizados

### 2.1 Modelo principal — SARIMAX

> Seasonal ARIMA con variable exógena (ventas de otros modelos Chery)

| Parámetro | Espacio de búsqueda | Mejor valor encontrado |
|-----------|--------------------|-----------------------|
| `p` (AR) | {0, 1, 2, 3} | 7* |
| `d` (diferenciación) | {0, 1} | 1 |
| `q` (MA) | {0, 1, 2, 3} | 2 |
| `P` (AR estacional) | {0, 1} | 1 |
| `D` (dif. estacional) | {0, 1} | 1 |
| `Q` (MA estacional) | {0, 1, 2} | 2 |
| `m` (periodo) | 12 (fijo) | 12 |

*p=7 excede el espacio definido: señal de overfitting o expansión no documentada del grid.

**Búsqueda de hiperparámetros:** Optuna TPE · 80 trials vs. 384 combinaciones en grid exhaustivo
- Velocidad: ~2 min (Optuna) vs. ~8 min (grid) → **4× más rápido**
- Criterio de optimización: MAPE en test out-of-sample

**Variable exógena:** `ventas_otros` (resto de modelos Chery del mismo mes)
- Justificación: captura efectos de mercado compartido (campañas, desabasto, estacionalidad de marca)
- Proyección futura: media móvil de 6 meses (workaround naive)

**Validación:** Walk-forward sobre los últimos 12 meses
- Por cada mes: reentrenamiento con datos históricos hasta ese punto → predicción 1 paso adelante
- Simula exactamente el comportamiento en producción

### 2.2 Modelos de comparativa (exploratorios)

| Modelo | Tipo | Feature engineering | Ventaja diferencial |
|--------|------|--------------------|--------------------|
| **Prophet** | Time series (Facebook) | Estacionalidad anual multiplicativa, festivos Perú | Robustez ante cambios de tendencia |
| **Regresión Lineal** | Supervisado | Lags 1,2,3,6,12 · rolling mean/std · mes, trimestre | Interpretabilidad de coeficientes |
| **Random Forest** | Ensemble | Ídem | Relaciones no lineales sin configuración explícita |
| **XGBoost** | Gradient Boosting | Ídem | Frecuentemente mejor MAPE; riesgo de overfitting en series cortas |

**Pipeline de features para modelos ML:**
```
serie → lags(1,2,3,6,12)
      → rolling_mean/std(3m, 6m) [shift=1 para evitar lookahead]
      → mes, trimestre
      → dropna() → train/test split
```

---

## 3. Resultados Obtenidos

### 3.1 Métricas del modelo ganador (SARIMA)

| Métrica | Valor | Referencia |
|---------|-------|-----------|
| **Orden** | (7, 1, 2)(1, 1, 2)[12] | — |
| **AIC** | 105.40 | Menor = mejor ajuste penalizado |
| **BIC** | 160.28 | Menor = menor complejidad |
| **MAPE walk-forward** | **27.89%** | Objetivo: <10% (excelente), <15% (aceptable) |
| **Predicción próximo mes** | ~20.1 unidades | Dato del último artefacto generado |

### 3.2 Diagnóstico del resultado

El MAPE de **27.89% está por encima del umbral aceptable**. Implica un error promedio de ±28 unidades sobre una media de ~65 unidades/mes actuales.

**Causas raíz identificadas:**

1. **Quiebre estructural en los datos:** demanda pasó de ~31 u/mes (2021–2023) a ~65 u/mes (2024–2026). El modelo "aprende" un régimen que ya no existe.

2. **Overfitting potencial:** p=7 en una serie de 62 observaciones es muy alto. El walk-forward mitiga pero no elimina el riesgo.

3. **Variable exógena ruidosa:** si `ventas_otros` tiene Pearson r < 0.3 con ventas Tiggo 2, introduce ruido en lugar de señal.

4. **Proyección naive del exógeno:** usar media constante para los meses del horizonte es demasiado simplista cuando el mercado es volátil.

---

## 4. Aprendizajes Clave

### Técnicos

| Aprendizaje | Detalle |
|-------------|---------|
| **Optuna TPE supera grid search** | No solo en velocidad (4×), sino en calidad: el muestreo bayesiano prioriza regiones prometedoras y evita combinaciones inútiles. Para hiperparámetros SARIMA con restricciones (no d=1 y D=1 simultáneo), Optuna maneja constraints nativamente. |
| **Walk-forward es la única validación válida en producción** | Train/test split ignora el efecto de reentrenamiento mensual. Walk-forward simula exactamente lo que ocurre en producción: cada mes se refit con los datos disponibles en ese momento. |
| **p alto no es necesariamente mejor** | p=7 en 62 observaciones viola la regla de parsimonia (mínimo ~10 obs por parámetro AR). Puede estar capturando ruido. |
| **La variable exógena necesita diagnóstico previo** | Antes de incluirla, verificar correlación estadística. Si r < 0.3, el modelo puro SARIMA será más robusto. |
| **El quiebre estructural invalida datos históricos** | Datos del período 2021–2023 introducen sesgo. Ventana de entrenamiento desde 2024-01 es la decisión correcta. |

### Funcionales / de Producto

| Aprendizaje | Detalle |
|-------------|---------|
| **La arquitectura modular pagó dividendos** | Separar `core/` (auth, I/O, validación, estilos) de las páginas permitió iterar el modelo sin tocar la interfaz y viceversa. |
| **RBAC desde el inicio es correcto** | Añadir roles post-facto es costoso. Definir 4 roles con 3 permisos granulares desde v1 simplificó el control de acceso en producción. |
| **El dual-write (DB + JSON) es deuda técnica aceptable** | Garantiza operación si Supabase falla, pero duplica la superficie de mantenimiento. En v2 debe resolverse con Supabase Realtime. |
| **El usuario de negocio necesita semáforos, no métricas** | MAPE 27.89% no dice nada a un gerente. El dashboard rojo/amarillo/verde es más accionable que el número. |
| **La validación de datos es preventiva, no opcional** | Los 17 test unitarios en `utils_validacion.py` detectaron 3 casos reales de data corruption durante el desarrollo. |

---

## 5. Puntos Clave para Slides Académicos

### Slide: Problema
> Chery Perú carece de un sistema cuantitativo para predecir demanda del Tiggo 2. Las decisiones de inventario y asignación a concesionarios son reactivas. **Objetivo:** reducir el error de pronóstico a <15% MAPE con una herramienta operacional usable por el equipo comercial.

### Slide: Arquitectura del Sistema
- **4 capas:** Datos → ML → Backend (Supabase) → Frontend (Streamlit)
- **Auth con RBAC:** 4 roles (Administrador, Analista, Gerencia, Concesionario)
- **Persistencia dual:** PostgreSQL para metadatos + Storage para artefactos (.pkl.gz)
- **Despliegue:** Streamlit Cloud (acceso web, sin instalación local)

### Slide: Modelos Evaluados
- **SARIMA/SARIMAX:** modelo principal, optimizado con Optuna TPE (80 trials)
- **Prophet:** alternativa para datos con quiebres de tendencia y festivos
- **ML supervisado:** Regresión Lineal, Random Forest, XGBoost con feature engineering manual
- **Decisión de diseño:** SARIMA como núcleo porque maneja autocorrelación temporal de forma nativa; ML como benchmark

### Slide: Resultados — Primera Iteración
| | SARIMA (ganador) |
|--|--|
| MAPE (walk-forward) | **27.89%** |
| AIC | 105.40 |
| Orden | (7,1,2)(1,1,2)[12] |
| Dataset | 62 meses |

> El modelo está **operativo pero no apto para decisión autónoma.** Se usa en modo diagnóstico con revisión humana obligatoria.

### Slide: Diagnóstico del MAPE Alto
1. Quiebre estructural 2021→2024 (×2 en volumen)
2. Serie corta: 62 observaciones limitan profundidad SARIMA
3. Exógeno proyectado con media constante (naive)
4. p=7 en 62 obs → posible overfitting

### Slide: Valor Demostrado — Arquitectura
- Sistema de autenticación robusto con fallback local
- Audit trail completo (quién aprobó, cuándo, con qué métricas)
- Versioning de modelos: solo 1 activo en producción, reversión en 1 clic
- Walk-forward validation como estándar de evaluación realista
- 17 tests unitarios de validación de datos

### Slide: Iteración 2 — Roadmap
| Prioridad | Acción | Impacto esperado |
|-----------|--------|-----------------|
| Alta | Reentrenar solo con datos 2024+ | Eliminar sesgo del quiebre estructural |
| Alta | Forzar p ≤ 3 en Optuna | Reducir overfitting |
| Alta | Validar Pearson(exógeno) antes de incluirlo | Evitar ruido en modelo |
| Media | Ensemble SARIMA + Prophet | Reducir MAPE ~5-8 pp |
| Media | Supabase Realtime (reemplazar polling) | Escalabilidad a 10+ usuarios |
| Baja | Quantile regression en modelos ML | Intervalos de confianza en Random Forest / XGBoost |

---

## 6. Veredicto Final

| Dimensión | Estado | Nota |
|-----------|--------|------|
| **Precisión del modelo** | ❌ Insuficiente | MAPE 27.89% vs. objetivo <15% |
| **Arquitectura del sistema** | ✅ Production-grade | Auth, RBAC, audit trail, versioning |
| **Pipeline de validación** | ✅ Correcto | Walk-forward replica producción exactamente |
| **Experiencia de usuario** | ✅ Operacional | Dashboard usable por perfiles no técnicos |
| **Calidad del código** | ✅ Mantenible | Modular, documentado, con tests |
| **Listo para producción** | ⚠️ Condicional | Sistema sí; modelo con supervisión humana |

> **Conclusión:** La Iteración 1 establece una base arquitectónica sólida y un pipeline de evaluación riguroso. El modelo no alcanza el objetivo de precisión, pero el sistema ya genera valor como herramienta de análisis y las causas del MAPE alto están diagnosticadas. La Iteración 2 tiene un roadmap técnico claro con alta probabilidad de alcanzar el umbral <15%.

---

*Sistema TIGGO 2 · ISDI Troncal · Diego Mijallodv · 2026*
