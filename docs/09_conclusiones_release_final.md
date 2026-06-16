# Conclusiones — Release Final

**Sistema TIGGO 2 · Predicción de Demanda · ISDI Troncal** _Fecha: 2026-06-16 · Versión del sistema: v40.1_

---

## 1. El Problema de Negocio

Int. Norte (distribuidor Chery en Perú) opera con un proceso de compra de vehículos basado en intuición y experiencia del equipo comercial. Sin datos estructurados ni predicciones formales:

- **Sobre-stock**: capital inmovilizado en vehículos que tardan en venderse.
- **Ruptura de stock**: demanda no atendida en meses pico — pérdida directa de venta.
- **Ciclo estacional ignorado**: el Tiggo 2 muestra patrones estacionales (picos en Q1 y Q3) que no se incorporan a las órdenes de compra.

**Hipótesis de valor:** si se puede predecir la demanda mensual con un error menor al 15%, la gerencia puede tomar mejores decisiones de compra con al menos 1–3 meses de anticipación.

---

## 2. La Solución

> *"Con los datos que ya tenías, pero con la inteligencia que antes no tenías."*

Sistema web multi-página construido sobre Streamlit + Supabase que combina predicción SARIMA con análisis de negocio, acceso por roles y un ciclo operativo completo.

### Arquitectura del sistema

```
[Datos históricos ventas Tiggo 2 + mercado Chery]
             ↓
    [Entrenamiento SARIMA — Optuna TPE]
             ↓
    [Validación Walk-Forward 12 meses]
             ↓
    [Dashboard — predicción 6 meses]
             ↓
    [Registro ventas reales → MAPE producción]
             ↓
    [Alerta proactiva si error > umbral → Re-entrenamiento]
```

**Stack:** Streamlit Cloud · Supabase (PostgreSQL + Storage) · statsmodels · Optuna · Prophet · scikit-learn · XGBoost · Google Gemini 2.5 Flash

---

## 3. Evolución del Modelo a lo Largo del Proyecto

| Iteración | MAPE Walk-Forward | Orden SARIMA | Estado |
|-----------|------------------|-------------|--------|
| **1** | 27.89% ❌ | (7,1,2)(1,1,2)\[12\] | Diagnóstico — modelo sobreparametrizado |
| **2** | **10.32% ✅** | **(1,1,0)(1,0,2)\[12\]** | Modelo óptimo |
| **3** | 14.65% ⚠️ | (2,0,1)(1,0,2)\[12\] | Sistema operativo completo |

> **Métrica de producción: MAPE Walk-Forward.** Se valida sobre los últimos 12 meses del dataset con esquema expanding window — en cada paso, el modelo se re-estima con datos hasta ese mes y predice el siguiente, sin data leakage. Esta es la única métrica que representa el error real en producción.

### Las decisiones técnicas que movieron el MAPE

| Iteración | Decisión clave | Impacto en MAPE |
|-----------|---------------|----------------|
| Iter1 → Iter2 | Recortar dataset a Ene 2022 — eliminar régimen bajo-volumen 2021 | ▼ 17.57 pp |
| Iter1 → Iter2 | Reducir p de 7 a 2, d=1→0 — eliminar overfitting | Incluido en ▼ 17.57 pp |
| Iter2 → Iter3 | Nuevos datos 2026 con tendencia alcista — dinámica cambia | ▲ 4.33 pp |
| Iter2 → Iter3 | Foco en producto (5 módulos nuevos) — no en optimización de modelo | — |

**Palanca principal:** la calidad del dataset importa más que la complejidad del modelo. Eliminar 12 meses de datos ruidosos (Iter1→Iter2) redujo el error ▼17.57 pp. La leve regresión en Iter3 refleja datos 2026 más volátiles, no un deterioro del pipeline.

> **El mejor modelo del proyecto es Iter2: MAPE 10.32%** — guardado en Supabase y activable en cualquier momento desde el panel de Administración.

---

## 4. El Sistema en Producción — 8 Módulos

| # | Módulo | Rol mínimo | Descripción |
|---|--------|-----------|-------------|
| 1 | **Entrenamiento** | Analyst | Carga datos, optimiza SARIMA con Optuna (80 trials), valida walk-forward, aprueba o rechaza modelo |
| 2 | **Dashboard** | Viewer | KPIs de demanda, predicción 6 meses, semáforo de confianza, ciclo estacional, alertas |
| 3 | **Comparativa ML** | Analyst | SARIMA vs Prophet vs XGBoost vs Random Forest — métricas comparativas, selección de modelo óptimo |
| 4 | **Concesionarios** | Manager | Predicciones desagregadas por punto de venta — planificación de distribución |
| 5 | **Proyección de Ingresos** | Manager | Proyección financiera 12 meses · ticket 15 000 USD · análisis de sensibilidad |
| 6 | **Escalabilidad** | Manager | Portafolio de 10 modelos con datos reales (Chery, Jetour, Ford, KIA, DFSK, Mitsubishi) · mapa de expansión LatAm · playbook de onboarding 2–4 semanas |
| 7 | **Registrar Ventas** | Analyst | Registro mensual de ventas reales · comparativa real vs predicho · MAPE de producción |
| 8 | **Administración** | Admin | Gestión de modelos activos, datos precargados en Supabase Storage, historial de entrenamientos |

---

## 5. Propuesta de Valor Cuantificada

### ROI estimado (escenario conservador — 1 concesionario piloto)

| Concepto | Cálculo | Ahorro estimado anual |
|---------|---------|----------------------|
| Reducción sobre-stock (5% del inventario) | 5 u/mes × 15 000 USD × 30% holding cost | **~27 000 USD** |
| Captura de demanda no atendida (2 u/mes) | 2 u/mes × 12 meses × margen ~2 000 USD | **~48 000 USD** |
| Reducción de tiempo en análisis de compra | 4 h/semana × 52 semanas × costo analista | **~10 400 USD** |
| **Total estimado** | | **~85 000 USD/año** |

*Supuestos conservadores. El sistema no reemplaza al comprador — le da información para decidir mejor.*

---

## 6. Modelos Descartados — Por Qué SARIMA Ganó

| Modelo | MAPE | Razón de descarte |
|--------|------|------------------|
| **SARIMAx** ✅ | **10.32%** | Mejor balance precisión / interpretabilidad |
| Prophet | ~18–22%* | Inflexible con series cortas (<60 obs); estacionalidad aditiva sobresuaviza los picos |
| XGBoost | ~24–30%* | Requiere >200 obs para generalizar bien; sin estructura temporal explícita |
| Random Forest | ~28–35%* | Mismo problema que XGBoost; alta varianza en series cortas |
| Regresión Lineal | ~35%* | No captura estacionalidad sin feature engineering manual extenso |

*Valores obtenidos en Comparativa ML con el mismo dataset y split de validación.

**Justificación de frecuencia de re-entrenamiento trimestral:**
- La demanda mensual del Tiggo 2 es estable dentro de un trimestre (coeficiente de variación ~18%).
- Re-entrenar semanalmente con 1–3 nuevos puntos no mejora el modelo; el ruido domina la señal.
- Re-entrenar trimestralmente incorpora la estacionalidad completa del trimestre anterior.
- El sistema lanza alerta automática si el MAPE de producción supera 15% antes del trimestre — en ese caso, se re-entrena manualmente.

---

## 7. Roadmap — Evolución del Producto

### Corto plazo (Q3 2026)
- Integrar datos AMDA (ventas del mercado total) como exógena adicional — se espera reducir MAPE a ~8%
- Alertas automáticas vía email/WhatsApp al área de compras cuando la predicción supera el stock disponible
- Dashboard embebido en el ERP existente de Int. Norte

### Mediano plazo (Q4 2026 – Q1 2027)
- Expandir a portafolio completo Chery Perú: Tiggo 4 PRO y Arrizo 5 listos para entrenar (≥ 51 meses de datos); Jetour X70 (sub-marca Chery) con 61 meses y 25 uds/mes — onboarding estimado 2–3 semanas
- Integrar CRM del distribuidor — correlacionar leads → conversiones → ventas para predicción leading
- Modelo de precios dinámicos basado en predicción de demanda y stock disponible

### Largo plazo (2027+)
- Escalabilidad regional: Chile, Colombia, Ecuador con el mismo stack
- Asistente IA proactivo — evolución de "responde preguntas" a "inicia conversaciones"
- Federación de modelos: un modelo global Chery LatAm con fine-tuning por país

### Contexto competitivo
Chery ocupa el **4.º lugar** en ventas de SUVs compactos en Perú (2025), detrás de Kia, Hyundai y Toyota. La ventaja competitiva no viene del precio — viene de la disponibilidad. Un sistema de predicción que garantice stock en los momentos de mayor demanda es un diferenciador operativo frente a distribuidores que operan sin inteligencia de inventario.

---

## 8. Aprendizajes del Proyecto

### Sobre modelos de series temporales

1. **El dataset limpio supera al modelo complejo.** Eliminar 12 meses de datos de régimen distinto redujo el error más que cualquier ajuste de hiperparámetros.
2. **La métrica de producción es walk-forward.** AIC/BIC miden el pasado; el MAPE walk-forward mide el futuro. En caso de conflicto, el MAPE gana siempre.
3. **La estacionariedad debe validarse en cada iteración.** Un modelo óptimo en un régimen puede requerir diferenciación al cambiar el contexto de datos.

### Sobre construcción de producto de datos

4. **La adopción depende de la UX, no del modelo.** El jurado de Release 2 valoró más el dashboard que la precisión del SARIMA. Un modelo perfecto sin interfaz no llega al usuario de negocio.
5. **El ciclo cerrado es el diferenciador.** Pasar de "sistema de predicción" a "sistema de decisión operativo" requirió conectar el registro de ventas reales con el módulo de re-entrenamiento. Ese lazo es lo que hace que el modelo no se degrade sin visibilidad.
6. **La infraestructura sí importa en producción.** Streamlit Cloud no tiene filesystem persistente. Resolver eso con Supabase Storage tomó tiempo en Iter3 pero es lo que permite que el sistema corra sin intervención manual en deploy.

### Sobre presentación académica

7. **Comenzar con el problema de negocio, no con el modelo.** Los primeros 60 segundos de cualquier presentación técnica deben ser de negocio: ¿quién pierde dinero, cuánto, y por qué este sistema lo resuelve?
8. **Los modelos descartados son evidencia de rigor.** Mostrar que se evaluaron 5 modelos y se eligió el mejor con criterios claros comunica más que presentar solo el modelo ganador.

---

## 9. Slide de Cierre — El Sistema en Una Vista

| | Iteración 1 | Iteración 2 | Iteración 3 (Release Final) |
|---|---|---|---|
| MAPE walk-forward | 27.89% ❌ | **10.32% ✅** | 14.65% ⚠️ |
| Páginas | 3 | 3 | **8** |
| Ciclo operativo | ❌ | ❌ | ✅ |
| Alertas proactivas | ❌ | ❌ | ✅ |
| Deploy sin archivos locales | ❌ | ❌ | ✅ |
| Escalabilidad multi-marca | ❌ | ❌ | ✅ |
| Estado | Diagnóstico | Modelo óptimo | **Sistema operativo completo** |

> **Conclusión:** En tres iteraciones, el Sistema TIGGO 2 evolucionó de un prototipo de predicción a un sistema de decisión operativo. El mejor modelo del proyecto (Iter2, MAPE 10.32%) está disponible en producción; el modelo de Iter3 cierra el ciclo con MAPE 14.65% — dentro del umbral operacional — y un sistema completo de 8 módulos con cobertura nacional, ciclo de retroalimentación automático y arquitectura escalable. Listo para un piloto real en Int. Norte.

---

---

## Correcciones post-Release Final

### v40.1 — 2026-06-16

- **fix(dashboard/alerta_predictiva)**: La alerta predictiva proactiva mostraba el mes incorrecto cuando el modelo no se había reentrenado recientemente. La causa era que usaba `pred_total.iloc[0]` (primer mes del horizonte, fijo desde el entrenamiento) en lugar del mes siguiente al calendario real. Con fecha 16 Jun 2026 y un modelo entrenado en abril, el sistema mostraba la predicción de Abril en vez de Julio. Corregido para usar la misma lógica de mes calendario que el KPI card del Dashboard.

---

_Sistema TIGGO 2 · ISDI Troncal · Diego · 2026_
