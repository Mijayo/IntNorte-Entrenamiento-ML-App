# Guía de Usuario

Esta guía explica cómo usar el sistema paso a paso, sin necesidad de conocimientos técnicos en programación o estadística. Está organizada por las tres secciones principales de la aplicación.

---

## Acceso al sistema

Al entrar a la URL de la aplicación se muestra el formulario de login. Introduce tu usuario y contraseña. Si introduces mal la contraseña tres veces, la cuenta se bloquea temporalmente.

La sesión expira automáticamente tras **60 minutos de inactividad**. El sistema te avisará y pedirá que vuelvas a iniciar sesión.

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
6. **Variable exógena** — se calcula el total mensual de ventas de los *otros* modelos de la misma marca. El sistema comprueba automáticamente si esta variable tiene correlación real con las ventas del modelo objetivo (Pearson r ≥ 0.3). Si la correlación es baja, la variable se descarta para no añadir ruido al modelo.

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

Tras pulsar **Iniciar Entrenamiento**, el sistema primero evalúa si la variable exógena (`ventas_otros`) tiene correlación suficiente con las ventas del modelo objetivo:

- `ℹ️ Variable exógena incluida — Pearson r = 0.61` → se usará en el modelo.
- `⚠️ Variable exógena descartada — Pearson r = 0.18` → SARIMA entrena sin ella.

El proceso tarda entre 1 y 3 minutos y muestra su progreso en tiempo real:

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

#### Admin (8 tabs en Dashboard + páginas independientes)

| Tab / Página | Descripción |
|---|---|
| 📊 Dashboard | KPIs: ventas del último mes, MAPE, horizonte. Gráfico del histórico completo. |
| 🔮 Predicciones | Histórico + predicción mes a mes con IC 95% + validación walk-forward. |
| 💼 Recomendaciones | Escenarios conservador y agresivo de compra al fabricante + marco teórico (Newsvendor Problem, stock de seguridad, nivel de servicio Tipo I). |
| 🔬 ACF/PACF | Gráficos de autocorrelación para interpretar la estructura de la serie. |
| 🔍 Grid Search | Resultados de la búsqueda Optuna: top modelos, scatter AIC vs MAPE. |
| 🔄 Walk-Forward | Real vs predicho mes a mes, tabla de errores porcentuales. |
| 📋 Métricas técnicas | Parámetros completos del modelo, AIC, BIC, residuos. |
| 🤖 Asistente IA | Chat con Gemini sobre el modelo (ver sección abajo). |
| 💰 **Proyección Ingresos** *(página independiente)* | Proyección financiera a 6 meses en dólares. |
| 🏪 **Concesionarios** *(página independiente)* | Análisis histórico + predicciones desagregadas por tienda. |

#### Analista (8 tabs en Dashboard + páginas independientes)

| Tab / Página | Descripción |
|---|---|
| 📊 Dashboard | KPIs: ventas del último mes, MAPE, horizonte. Gráfico del histórico completo. |
| 🔮 Predicciones | Histórico + predicción mes a mes con IC 95% + validación walk-forward. |
| 💼 Recomendaciones | Escenarios conservador y agresivo de compra al fabricante + marco teórico académico. |
| 🔬 ACF/PACF | Gráficos de autocorrelación para interpretar la estructura de la serie. |
| 🔍 Grid Search | Resultados de la búsqueda Optuna: top modelos, scatter AIC vs MAPE. |
| 🔄 Walk-Forward | Real vs predicho mes a mes, tabla de errores porcentuales. |
| 📋 Métricas técnicas | Parámetros completos del modelo, AIC, BIC, residuos. |
| 🤖 Asistente IA | Chat con Gemini sobre el modelo (ver sección abajo). |
| 💰 **Proyección Ingresos** *(página independiente)* | Proyección financiera a 6 meses en dólares. |
| 🏪 **Concesionarios** *(página independiente)* | Análisis histórico + predicciones desagregadas por tienda. |

#### Financiero (2 tabs en Dashboard + página Proyección Ingresos)

> Disponible para: `financiero`

| Tab / Página | Descripción |
|---|---|
| 📊 Dashboard | KPIs e histórico básico. |
| 🔮 Predicciones | Predicciones mes a mes con IC 95% + validación walk-forward. |
| 💰 **Proyección Ingresos** *(página independiente)* | Proyección financiera a 6 meses en dólares con exportación CSV. |

#### Gerente (4 tabs en Dashboard + página Concesionarios)

| Tab / Página | Descripción |
|-----|-------------|
| 📊 Dashboard | KPIs e histórico. |
| 🔮 Predicciones | Predicciones mes a mes con IC 95% + validación walk-forward. |
| 💼 Recomendaciones | Escenarios conservador (+10%) y agresivo (+20%) de compra al fabricante, análisis de tendencia y expander con marco teórico académico. |
| 🤖 Asistente IA | Chat orientado a decisiones de negocio. |
| 🏪 **Concesionarios** *(página independiente)* | Análisis histórico + predicciones desagregadas por tienda. |

#### Viewer (2 tabs)

Dashboard básico y predicciones, sin métricas técnicas ni IA ni proyección financiera.

---

### Tab Recomendaciones de Compra

> Disponible para: `admin`, `analyst`, `manager`

Muestra el análisis de demanda del próximo mes y dos escenarios de pedido al fabricante.

**Métricas mostradas:**
- Predicción puntual y rango IC 95% para el mes siguiente.
- **Estrategia Conservadora** — IC superior + 10%: minimiza sobrestock; recomendada cuando la tendencia es estable o decreciente.
- **Estrategia Agresiva** — IC superior + 20%: maximiza cobertura; recomendada cuando la tendencia es creciente o el lead time de importación es largo.
- **Señal de tendencia** — comparación de los últimos 3 meses vs el promedio histórico (umbral ±10%).

**Expander "📚 Marco teórico — ¿Por qué estas estrategias?"** *(disponible para profundizar)*

Explica el fundamento académico de cada recomendación:

| Concepto | Referencia |
|----------|-----------|
| Newsvendor Problem | Scarf (1958) — uso del percentil alto como base cuando el coste de rotura supera al de sobrestock |
| Stock de seguridad (+10%) | Silver, Pyke & Thomas (1998) — `SS = z · σ_L` para absorber error residual del modelo |
| Nivel de servicio Tipo I (+20%) | Zipkin (2000) — cubre ~99% de los escenarios cuando lead time > 60 días |
| Tabla de limitaciones | Condiciones de validez: gaussianidad, estacionariedad, MAPE < 15% |

---

### Tab Predicciones — dos conceptos clave

> Es importante entender la diferencia entre los dos escenarios temporales del modelo.

**① Predicción mes a mes**
El modelo genera una estimación *independiente* para cada mes del horizonte. Cada fila de la tabla es una predicción propia con su intervalo de confianza al 95% — no es el total de 6 meses dividido entre 6.

**② Horizonte de 6 meses**
Es la ventana de visibilidad hacia adelante. En operación real, el equipo actualiza el histórico cada mes con las ventas cerradas y relanza el modelo; el horizonte te permite ver más lejos, pero la predicción más fiable es siempre la del mes inmediato.

La **zona violeta** del gráfico muestra la validación walk-forward: simula exactamente ese proceso de renovación mensual — el modelo predijo cada mes un solo paso adelante con todos los datos anteriores disponibles. El KPI "MAPE real (1 mes)" refleja ese error real de producción (objetivo < 15%).

---

### Página Proyección de Ingresos

> Disponible para: `admin`, `analyst`, `financiero` (permiso `ver_ingresos`)

Página independiente que traduce la predicción SARIMA en cifras financieras en dólares. Los roles sin el permiso `ver_ingresos` (gerente, viewer) verán un aviso de acceso restringido al intentar entrar.

**Parámetros configurables:**

| Campo | Por defecto | Descripción |
|-------|-------------|-------------|
| Precio medio por unidad (USD $) | 27 000 $ | Precio neto de venta por vehículo en dólares |
| Margen neto estimado (%) | 8 % | Porcentaje de beneficio neto; pon 0 para omitirlo |
| Tipo de cambio | 1.00 | Multiplica el precio para convertir a moneda local si aplica |

**Lo que muestra:**

- **KPIs de resumen**: unidades totales predichas, ingresos totales estimados a 6 meses y rango de incertidumbre IC 95% en dólares.
- **KPIs de beneficio** (si margen > 0): beneficio estimado total y margen aplicado.
- **Gráfico de barras**: ingresos proyectados mes a mes con la banda IC 95% superpuesta en semitransparente. Si el margen es > 0, una línea adicional muestra el beneficio neto mensual.
- **Tabla mes a mes**: predicción en unidades · ingresos · IC inferior/superior en $ · beneficio (si aplica) · fila de totales al final.

> Los ingresos son el producto de las unidades predichas × precio × tipo de cambio. El rango IC 95% se traslada a dólares para comunicar la incertidumbre financiera de forma comprensible para cualquier rol. Los roles con permiso `exportar` verán el botón **Exportar CSV proyección USD**.

---

### Página Concesionarios

> Disponible para: `admin`, `analyst`, `manager`

Página independiente que combina el análisis histórico de ventas por tienda con predicciones SARIMA desagregadas. Para comenzar, carga el Excel de ventas desde el expander **📂 Cargar datos de ventas**. El sistema normaliza los nombres de columnas automáticamente (acepta `FECHA-VENTA`, `FECHA_VENTA`, `FECHA VENTA`, `CONCESIONARIO`, `DET_CC`, `AGE` o `SUCURSAL`).

**Filtros disponibles (barra lateral):** año, modelo de vehículo, concesionarios a incluir.

**Tab 📊 Resumen:** barras horizontales de ventas totales por tienda con porcentaje del total + gráfico de mix de modelos apilado por concesionario.

**Tab 📈 Evolución Mensual:** evolución mensual con líneas por tienda, gráfico de share de mercado mensual en área 100% apilada, y barras de variación MoM agrupada por concesionario.

**Tab 🔮 Predicciones por Tienda:**
- **Metodología:** el modelo SARIMA predice el total nacional. El share de los últimos 12 meses de cada concesionario se usa para repartir esa predicción (y sus IC 95%) entre las tiendas.
- **Editor de shares:** expander que permite ajustar el % de cada tienda para simular escenarios (apertura/cierre, campaña local). Los shares se renormalizan automáticamente.
- **KPIs próximo mes:** una tarjeta por concesionario con las unidades predichas y el rango IC 95%.
- **Gráfico histórico + predicción:** líneas de histórico real continuadas con línea punteada de predicción y banda IC 95% por tienda.
- **Barras de horizonte:** barras apiladas para todos los meses del horizonte con hover de IC 95%.
- **Tabla de predicciones:** exportable vía CSV para roles con permiso `exportar`.

**Tab 📋 Detalle:** ranking acumulado (% y % acumulado) + pivot mensual exportable.

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
