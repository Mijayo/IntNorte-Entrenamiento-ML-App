# Página 2 — Comparativa de Modelos ML

**Archivo:** `pages/2_Comparativa_ML.py`  
**Acceso:** `admin`, `analyst` (permiso `entrenar_modelos`)  
**Icono:** 🏆

---

## Propósito

Compara hasta **5 modelos** de predicción sobre el mismo histórico mensual del Tiggo 2 y determina cuál genera menor error (MAPE). Permite publicar el run ganador en producción directamente desde esta página.

**Modelos disponibles:**

| Modelo | Familia | Variable exógena |
|--------|---------|-----------------|
| SARIMAX | Serie temporal | ✅ `ventas_otros` (igual que en Entrenamiento) |
| Prophet | Serie temporal | ❌ no incorporada |
| Regresión Lineal | ML — lag features | Como feature |
| Random Forest | ML — lag features | Como feature |
| XGBoost | ML — lag features | Como feature |

---

## Flujo de uso

### Paso 1 — Fuente de datos

Dos opciones seleccionables con radio button:

**Opción A — Run guardado en Supabase** *(recomendado)*

- Selecciona un run existente.
- El histórico y la variable exógena se cargan automáticamente.
- Los parámetros SARIMA se auto-completan desde las métricas del run.
- Si el run no tiene exógena (generado con versión anterior), SARIMAX corre sin ella.

**Opción B — Excel manual**

- Sube un `.xlsx` con columnas de fecha y ventas.
- Selecciona la columna de fecha, la de ventas y, opcionalmente, una variable exógena para SARIMAX.

---

### Paso 2 — Período de análisis

Selecciona el rango de meses a usar mediante dos selectboxes. El período activo se muestra en el caption con el conteo de meses incluidos.

---

### Paso 3 — Configuración

| Parámetro | Por defecto | Descripción |
|-----------|:-----------:|-------------|
| Meses para test (hold-out) | 6 | Últimos N meses reservados para medir el error |
| Modelos a comparar | Todos | Checkboxes individuales por modelo |
| Parámetros SARIMAX | Auto | p, d, q, P, D, Q — autocompletan desde el run cargado |
| Festivos Perú (Prophet) | ✅ Sí | Añade festivos peruanos al modelo Prophet |

> **Requisito mínimo ML:** los modelos de lag features necesitan ≥ `12 + n_test + 5` meses de histórico. Si no hay suficientes datos, se muestra advertencia y el botón queda deshabilitado.

---

### Paso 4 — Ejecutar comparación

Pulsa **🏆 Comparar modelos**. El sistema entrena cada modelo seleccionado en secuencia, mostrando barra de progreso.

**Feature engineering para modelos ML (`crear_features`):**

- Lags: 1, 2, 3, 6, 12 meses.
- Rolling media y std sobre 3 y 6 meses (desplazados 1 mes para evitar data leakage).
- Features de calendario: mes, trimestre.
- Se eliminan las primeras 12 filas con NaN.

---

### Paso 5 — Resultados

**Tabla de métricas** con celdas resaltadas (mínimo = verde para MAE, RMSE, MAPE; máximo = verde para R²; tiempo mínimo = amarillo):

| Métrica | Significado | Objetivo |
|---------|-------------|---------|
| MAPE (%) | Error % promedio sobre el valor real | Menor |
| MAE | Error promedio en unidades vendidas | Menor |
| RMSE | Como MAE pero penaliza errores grandes | Menor |
| R² | Proporción de varianza explicada (1.0 = perfecto) | Mayor |

> **Nota de cálculo:** MAPE usa denominador `(real + 0.1)` para evitar división por cero en meses con ventas nulas — idéntico al criterio del modelo de producción.

**Banner de ganador:**
```
🏆 Mejor modelo (MAPE): [nombre] — X.X% · N.Npp mejor que [segundo]
```

**Gráficas adicionales:**
- Predicciones vs real en el período de test (líneas por modelo).
- Error absoluto por mes y modelo (barras agrupadas).
- Importancia de features para modelos ML (barras agrupadas de importancia Gini / coeficiente absoluto).
- Tabla detallada mes a mes con predicción de cada modelo y error.

**Expander "📚 ¿Cómo interpretar estos resultados?"** — explica:
- Umbral de referencia: MAPE < 10% excelente, < 20% aceptable en ventas automotrices.
- Por qué Prophet puede ganar (estacionalidad multiplicativa, changepoints, festivos).
- Cuándo SARIMA puede ganar (series cortas, variable exógena con correlación alta).
- Por qué los modelos ML pueden ganar o fallar (no-linealidad vs. extrapolación de tendencia).

---

### Paso 6 — Publicar modelo ganador en producción

Si la fuente de datos fue un run de Supabase, aparece el botón:

**✅ Activar `[run_label]` en el Dashboard**

Al pulsarlo:
- El run se marca como activo (`activo=TRUE` en Supabase).
- Los resultados de la comparativa (métricas de los 5 modelos y el ganador) se guardan en Supabase para mostrarse en el sub-tab **🏆 vs Descartados** del Dashboard sin necesidad de re-ejecutar.
- El Dashboard refleja el cambio de inmediato.

> Si el MAPE del ganador es > 20%, aparece una advertencia recomendando reentrenar antes de publicar.

---

## Paleta de colores por modelo

| Modelo | Color |
|--------|-------|
| SARIMAX | `#1C7293` (azul oscuro) |
| Prophet | `#E84855` (rojo) |
| Reg. Lineal | `#2ECC71` (verde) |
| Random Forest | `#F39C12` (naranja) |
| XGBoost | `#9B59B6` (violeta) |

---

## Funciones internas

| Función | Descripción |
|---------|-------------|
| `crear_features(series)` | Genera DataFrame con lags y rolling para modelos ML |
| `calc_metrics(real, pred, label)` | Calcula MAE, RMSE, MAPE, R² para un modelo |
| `entrenar_sarima(train, test_len, order, seasonal_order, ...)` | Entrena SARIMAX y devuelve array de predicciones clipeado a [0, ∞) |
| `entrenar_prophet(train, test_len, usar_holidays)` | Entrena Prophet con estacionalidad multiplicativa |
| `entrenar_ml(series, n_test, ModelClass, **kwargs)` | Entrena cualquier modelo sklearn con lag features |
| `plot_predicciones(train, test, predicciones)` | Gráfico Plotly de histórico + real + predicciones de todos los modelos |
| `plot_errores(test, predicciones)` | Barras de error absoluto por mes |
| `plot_importancias(importancias)` | Barras de feature importance para modelos ML |
