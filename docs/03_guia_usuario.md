# Guía de Usuario

Esta guía explica cómo usar el sistema paso a paso, sin necesidad de conocimientos técnicos en programación o estadística. Está organizada por las tres secciones principales de la aplicación.

---

## Acceso al sistema

Al entrar a la URL de la aplicación se muestra el formulario de login. Introduce tu usuario y contraseña. Si introduces mal la contraseña tres veces, la cuenta se bloquea temporalmente.

La sesión expira automáticamente tras **30 minutos de inactividad**. El sistema te avisará y pedirá que vuelvas a iniciar sesión.

---

## Sección 1 — Entrenamiento de Modelos

> Disponible para: `admin`, `analyst`

Esta sección está dividida en seis pestañas que deben seguirse **en orden**.

---

### Pestaña 1 — Cargar Datos

**Objetivo:** Subir el Excel con el histórico de ventas.

**Pasos:**
1. Haz clic en **Browse files** y selecciona uno o varios archivos `.xlsx`.
2. El sistema detecta automáticamente el tipo de archivo según la hoja que contiene:
   - Hoja `Hoja1` → archivo de ventas
   - Hoja `Stock Actual` → archivo de stock
3. Haz clic en **Procesar**.

El sistema aplica limpieza automática:
- Elimina filas donde el campo `MODELO3` está vacío (registros sin modelo asignado).
- Elimina ventas duplicadas del mismo vehículo (mismo número de chasis), conservando el registro más reciente.

Al finalizar, verás un resumen como:

```
✅ ventas_2024.xlsx procesado — 8,432 filas cargadas
   └─ 12 filas eliminadas por MODELO3 nulo
   └─ 34 duplicados por CHASIS eliminados
   └─ 8,386 filas limpias listas para validar
```

> **Consejo:** Si el histórico está dividido en varios archivos (por año, por ejemplo), puedes subirlos todos a la vez — el sistema los unifica automáticamente.

---

### Pestaña 2 — Validación

**Objetivo:** Verificar que los datos tienen la calidad suficiente para entrenar un modelo.

El sistema ejecuta los siguientes checks automáticamente:

| Check | ¿Qué verifica? |
|-------|----------------|
| Columnas requeridas | Que existan `FECHA-VENTA`, `MODELO3` y `MARCA` |
| Fechas válidas | Que menos del 5% de las fechas sean inválidas o nulas |
| Período mínimo | Que haya al menos **36 meses** de datos |
| Datos faltantes | Que ninguna columna clave supere el 5% de nulos |
| Outliers | Que no haya meses con ventas inusualmente altas o bajas (regla 3σ) |

Los errores **bloqueantes** (columnas faltantes, período insuficiente) impiden continuar. Las **advertencias** (outliers, algunos nulos) permiten continuar pero se muestran en amarillo para que las evalúes.

---

### Pestaña 3 — Preparar Datos *(académico)*

**Objetivo:** Entender, paso a paso, cómo se transforma el Excel bruto en la serie temporal que usa el modelo.

Esta pestaña no entrena nada — es puramente informativa. Muestra seis pasos:

1. **Datos brutos** — muestra las primeras filas del Excel tal como se cargó.
2. **Filtro marca** — solo se conservan las filas de la marca configurada (ej. CHERY).
3. **Filtro modelo** — solo se conservan las filas del modelo objetivo (ej. TIGGO 2).
4. **Rango de fechas** — se recorta el período según las fechas configuradas.
5. **Resample mensual** — se cuenta cuántas ventas hubo cada mes, generando la serie temporal.
6. **Variable exógena** — se calcula el total mensual de ventas de los *otros* modelos de la misma marca, que se usará como señal adicional en el modelo.

Al final hay un botón para **descargar el Excel de entrenamiento** (tres hojas: `Serie_SARIMA`, `Ventas_Mensuales`, `Comparativa`), útil para revisiones externas o documentación.

---

### Pestaña 4 — Entrenamiento

**Objetivo:** Configurar y lanzar el entrenamiento del modelo.

**Parámetros configurables:**

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| Marca | CHERY | Filtro de marca para construir la serie |
| Modelo | TIGGO 2 | Modelo de vehículo a predecir |
| Fecha inicio | 2021-01-01 | Ignora ventas anteriores a esta fecha |
| Fecha fin | Hoy | Límite superior del histórico |
| Excluir mes actual | Sí | Elimina el mes en curso (datos incompletos) |
| Horizonte | 6 meses | Cuántos meses hacia adelante predecir |
| Máx. ventas esperadas | 100 unid./mes | Descarta combinaciones que predicen valores irracionales |

Haz clic en **Iniciar Entrenamiento**. El proceso tarda entre 1 y 3 minutos y muestra su progreso en tiempo real:

```
🔍 Optuna trial 45/80 · Evaluados: 45 · Válidos: 32 · Descartados: 13 · Mejor: MAPE 4.87%
```

**¿Qué significa cada contador?**

- **Evaluados**: combinaciones de parámetros que Optuna ha probado hasta ahora.
- **Válidos**: combinaciones cuyas predicciones son coherentes (entre 0 y el máximo configurado) y cuyo error se pudo medir.
- **Descartados**: combinaciones que se rechazaron porque el modelo divergió, produjo predicciones negativas o superiores al límite, o falló por razones numéricas. *No indican un problema* — son un filtro de calidad normal en la búsqueda de hiperparámetros.

Al terminar verás un resultado como:

```
✅ Mejor modelo: SARIMA(1,1,1)(1,1,1,12)
   AIC: 450.23 · MAPE (Optuna test): 4.87% · 67 válidos / 13 descartados

✅ MAPE walk-forward: 5.34% · 6/6 meses validados
✅ Guardado en Supabase como 20260325_143000. Ve a Comparación para activarlo.
```

---

### Pestaña 5 — Comparación

**Objetivo:** Decidir si el nuevo modelo es mejor que el que está en producción.

Se muestran las métricas clave lado a lado:

| Métrica | Modelo actual | Modelo nuevo | Δ |
|---------|:-------------:|:------------:|---|
| MAPE walk-forward | 6.12% | 5.34% | -0.78% ✅ |
| AIC | 465.10 | 450.23 | -14.87 ✅ |
| Predicción próximo mes | 43 uds | 45 uds | +2 |

Si el nuevo modelo mejora en MAPE y AIC, aparece la recomendación:

```
✅ El nuevo modelo es mejor. Se recomienda aprobarlo.
```

Para activarlo en producción haz clic en **Aprobar y activar en Dashboard**. El Dashboard mostrará inmediatamente las predicciones del nuevo modelo.

> Los runs no aprobados quedan almacenados en el historial — no se borran y se pueden consultar desde el Dashboard seleccionándolos en la barra lateral.

---

### Pestaña 6 — Historial

**Objetivo:** Consultar todos los entrenamientos pasados.

Muestra una tabla con cada run: fecha, usuario, parámetros SARIMA encontrados, AIC y MAPE. Incluye un gráfico de evolución del MAPE a lo largo del tiempo para ver si los modelos han mejorado.

---

## Sección 2 — Dashboard

> Disponible para todos los roles

El Dashboard carga automáticamente el modelo activo al arrancar. Desde la barra lateral puedes cambiar a cualquier run histórico sin afectar producción (el cambio es solo visual, para tu sesión).

---

### Tabs disponibles según rol

#### Admin y Analista (8 tabs)

| Tab | Descripción |
|-----|-------------|
| 📊 Dashboard | KPIs: ventas del último mes, MAPE, horizonte. Gráfico del histórico completo. |
| 🔮 Predicciones | Histórico + predicción de los próximos N meses con banda de confianza al 95%. |
| 🔬 ACF/PACF | Gráficos de autocorrelación para interpretar la estructura de la serie. |
| 🔍 Grid Search | Resultados de la búsqueda Optuna: top modelos, scatter AIC vs MAPE. |
| 🔄 Walk-Forward | Real vs predicho mes a mes, tabla de errores porcentuales. |
| 📋 Métricas técnicas | Parámetros completos del modelo, AIC, BIC, residuos. |
| 🤖 Asistente IA | Chat con Gemini sobre el modelo (ver sección abajo). |
| 🏪 Concesionarios | Análisis de ventas por distribuidor (requiere cargar Excel en el sidebar). |

#### Gerente (5 tabs)

| Tab | Descripción |
|-----|-------------|
| 📊 Dashboard | KPIs e histórico. |
| 🔮 Predicciones | Predicciones con intervalos de confianza. |
| 💼 Recomendaciones | Escenarios conservador y agresivo de compra al fabricante. |
| 🤖 Asistente IA | Chat orientado a decisiones de negocio. |
| 🏪 Concesionarios | Análisis por distribuidor. |

#### Viewer (2 tabs)

Dashboard básico y predicciones, sin métricas técnicas ni IA.

---

### Tab Concesionarios

Requiere subir un Excel desde el expander **📂 Datos de Concesionarios** en la barra lateral. El sistema normaliza automáticamente los nombres de columnas (acepta `FECHA-VENTA`, `FECHA_VENTA` o `FECHA VENTA`, por ejemplo).

Los filtros disponibles son: año, modelo de vehículo y ciudad.

Las visualizaciones incluyen:
- Barras horizontales por concesionario, coloreadas por ciudad
- Evolución mensual de los 5 concesionarios con más ventas
- Barras apiladas de modelos por concesionario
- Tabla de ranking con % acumulado (análisis ABC)

---

### Tab Asistente IA

El asistente conoce el modelo activo: parámetros SARIMA, MAPE, predicciones y tendencia reciente. Puedes hacerle preguntas en lenguaje natural.

**Ejemplos de preguntas (rol técnico):**
- *"¿Por qué el MAPE del walk-forward es mayor que el del test Optuna?"*
- *"¿Qué significa que P=1 en el componente estacional?"*
- *"¿Es un buen modelo con un AIC de 450?"*

**Ejemplos de preguntas (rol gerente):**
- *"¿Debería pedir más o menos unidades que el mes pasado?"*
- *"¿Cuál es el rango realista de ventas para abril?"*
- *"La predicción baja en verano, ¿es normal para este modelo?"*

---

## Sección 3 — Prophet vs SARIMA

> Disponible para: `admin`, `analyst`

Esta sección permite comparar académicamente los dos modelos sobre el mismo histórico.

**Paso 1 — Fuente de datos:**
- Opción A: Seleccionar un run ya guardado en Supabase (usa el histórico de ese entrenamiento).
- Opción B: Subir un Excel propio con columnas de fecha y ventas.

**Paso 2 — Configuración:**

| Parámetro | Descripción |
|-----------|-------------|
| Meses de test (hold-out) | Cuántos meses finales se reservan para medir el error real |
| Festivos México | Activa la incorporación de días festivos mexicanos en Prophet |
| p, d, q, P, D, Q | Parámetros SARIMA que se usarán en la comparación (no hay búsqueda automática aquí) |

**Paso 3 — Resultados:**

Se muestra una tabla comparativa de métricas y el modelo ganador según MAPE:

```
🏆 Prophet gana por MAPE: 3.21% vs 5.34% (SARIMA)
```

O si la diferencia es pequeña:

```
🤝 Empate técnico (diferencia < 1pp): Prophet 4.80% vs SARIMA 5.10%
```

También se incluye una **explicación didáctica automática** que interpreta por qué uno superó al otro en este conjunto de datos concreto.
