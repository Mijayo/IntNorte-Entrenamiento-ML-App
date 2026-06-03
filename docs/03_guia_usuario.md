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

**Objetivo:** Cargar el histórico de ventas para entrenar el modelo.

La pestaña ofrece dos modos de carga seleccionables con un radio button:

#### Modo A — 📦 Datos precargados (recomendado)

El sistema carga automáticamente los archivos que ya están en el servidor:

| Archivo | Contenido |
|---------|-----------|
| `data/processed/veh_ml_features.xlsx` | Histórico de ventas Ene 2017 – Mar 2026 (~30 039 registros, incluye `MODELO3`) |
| `data/raw/Stock Vehiculos.xlsx` | Stock actual del concesionario |

**Pasos:**
1. Deja seleccionado **📦 Datos precargados** (opción por defecto).
2. Haz clic en **✅ Cargar**.

La primera carga lee el disco; las siguientes son instantáneas (caché en memoria).

#### Modo B — 📤 Subir nuevo Excel

Para actualizar los datos con un histórico más reciente o diferente:

1. Haz clic en **Browse files** y selecciona uno o varios archivos `.xlsx`.
2. El sistema detecta automáticamente el tipo de archivo según la hoja:
   - Hoja `Hoja1` → archivo de ventas
   - Hoja `Stock Actual` → archivo de stock
3. Haz clic en **Procesar**.

> **Consejo:** Si el histórico está dividido en varios archivos, puedes subirlos todos a la vez — el sistema los unifica automáticamente.

#### Limpieza automática (ambos modos)

- Elimina filas donde `MODELO3` está vacío (registros sin modelo asignado).
- Elimina ventas duplicadas del mismo chasis, conservando el registro más reciente.

Al finalizar, verás un resumen con el total de registros en memoria y las columnas disponibles.

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

#### Admin / Analista (6 tabs en Dashboard + páginas independientes)

| Tab / Página | Descripción |
|---|---|
| 📊 Dashboard | KPIs: ventas del último mes, MAPE, horizonte. Gráfico del histórico completo. |
| 🔮 Predicciones | Histórico + predicción mes a mes con IC 95% + validación walk-forward. |
| 💼 Recomendaciones | Análisis del ciclo estacional (mes pico/valle, ratio, gráfico mensual, efecto rappel) + escenarios conservador y agresivo de compra al fabricante + marco teórico (Newsvendor Problem, stock de seguridad, nivel de servicio Tipo I). |
| 🔄 Walk-Forward | Real vs predicho mes a mes, tabla de errores porcentuales. |
| 📋 Métricas Técnicas | Cuatro sub-pestañas: **📊 Resumen** (parámetros SARIMA, AIC/BIC, MAPE), **🔬 ACF/PACF** (gráficos de autocorrelación), **🔍 Grid Search** (top modelos Optuna, scatter AIC vs MAPE), **🏆 vs Descartados** (comparativa de los 5 modelos con justificación de descarte y métricas en tiempo real si se viene de Comparativa ML). |
| 🤖 Asistente IA | Chat con Gemini sobre el modelo (ver sección abajo). |
| 💰 **Proyección Ingresos** *(página independiente)* | Proyección financiera a 6 meses en dólares + calculadora de ROI estratégico. |
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

Muestra el análisis del ciclo estacional, el análisis de demanda del próximo mes y dos escenarios de pedido al fabricante.

**Bloque 1 — Ciclo estacional:**

El sistema calcula automáticamente el patrón estacional a partir de los datos históricos reales del run activo:

- **KPIs**: mes con mayor demanda histórica promedio, mes con menor demanda, y ratio pico/valle (ej. "Diciembre es 2.3× más alto que Agosto").
- **Gráfico de barras mensual**: media histórica de ventas por mes del año. El mes máximo aparece en rojo, los meses sobre la media en azul, los meses bajo la media en gris.
- **Callout de negocio**: explica el efecto rappel del proveedor en diciembre (pico artificial de demanda por incentivos de fin de año) y la oportunidad operativa de anticipar el pedido en noviembre para aprovechar ese rappel.

**Bloque 2 — Recomendaciones de compra:**
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
| Precio medio por unidad (USD $) | 15 000 $ | Precio neto de venta por vehículo en dólares |
| Margen neto estimado (%) | 8 % | Porcentaje de beneficio neto; pon 0 para omitirlo |
| Tipo de cambio | 1.00 | Multiplica el precio para convertir a moneda local si aplica |

**Lo que muestra:**

- **KPIs de resumen**: unidades totales predichas, ingresos totales estimados a 6 meses y rango de incertidumbre IC 95% en dólares.
- **KPIs de beneficio** (si margen > 0): beneficio estimado total y margen aplicado.
- **Gráfico de barras**: ingresos proyectados mes a mes con la banda IC 95% superpuesta en semitransparente. Si el margen es > 0, una línea adicional muestra el beneficio neto mensual.
- **Tabla mes a mes**: predicción en unidades · ingresos · IC inferior/superior en $ · beneficio (si aplica) · fila de totales al final.

> Los ingresos son el producto de las unidades predichas × precio × tipo de cambio. El rango IC 95% se traslada a dólares para comunicar la incertidumbre financiera de forma comprensible para cualquier rol. Los roles con permiso `exportar` verán el botón **Exportar CSV proyección USD**.

---

### Calculadora de ROI estratégico

Al final de la página hay una sección interactiva: **"Valor Estratégico del Sistema — ¿Cuánto vale predecir bien?"**. Permite cuantificar el retorno económico del sistema en términos de ahorro de capital inmovilizado y ventas recuperadas.

**Cómo usarla:**

1. Introduce el sobrestock mensual actual (unidades medias en exceso por mes).
2. Indica el costo de financiamiento mensual (% sobre el valor de inventario).
3. Introduce los stockouts mensuales estimados (ventas perdidas por rotura de stock).
4. Ajusta el porcentaje de mejora esperado para sobrestock y stockout con el sistema.
5. Introduce el costo anual de operar el sistema (licencias, cómputo, soporte).

El sistema calcula automáticamente:

- **Ahorro por sobrestock** = sobrestock × reducción × costo financiamiento × precio × 12 meses.
- **Ingresos recuperados** = stockouts × reducción × margen × precio × 12 meses.
- **Valor neto anual** = valor bruto − costo del sistema.
- **ROI multiplier**: cuántas veces el costo del sistema se recupera en valor generado.
- **Payback en meses**: en cuántos meses el sistema paga su propia inversión.

**Gráfico waterfall**: muestra el flujo completo — desde los dos bloques de ahorro hasta el descuento del costo, llegando al valor neto final.

**Tablas comparativas**: "✅ Con sistema" vs "❌ Sin sistema" con el impacto económico de cada escenario.

---

### Página Concesionarios

> Disponible para: `admin`, `analyst`, `manager`

Página independiente que combina el análisis histórico de ventas por tienda con predicciones SARIMA desagregadas. La página carga automáticamente `data/raw/Historico_Ventas.xlsx` al arrancar (sin necesidad de acción del usuario). El expander **📂 Fuente de datos** muestra un badge con el origen activo y el número de registros CHERY disponibles.

**Cargar un Excel personalizado:** despliega el expander **📂 Fuente de datos**, selecciona un archivo `.xlsx`/`.xls` y el sistema lo normaliza automáticamente (acepta `FECHA-VENTA`, `FECHA_VENTA`, `FECHA VENTA`, `CONCESIONARIO`, `DET_CC`, `AGE` o `SUCURSAL`). El archivo personalizado reemplaza los datos precargados durante toda la sesión. Para volver al precargado, haz clic en **↩ Usar precargados**.

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

## Sección — Registrar Ventas Reales

> Disponible para: `admin`, `analyst`

Página de feedback loop. Una vez cerrado un mes, registra las ventas reales para medir la precisión del modelo en producción.

**Pasos:**

1. Selecciona el mes y el año en el selectbox.
2. Introduce el número de unidades vendidas ese mes.
3. Haz clic en **Registrar**. El dato se guarda en Supabase y el Dashboard se actualiza automáticamente.

**Scoreboard acumulado:**

Una vez hay al menos un mes registrado, la página muestra:
- MAPE de producción: error medio entre predicción y real, para todos los meses registrados.
- Desviación máxima y porcentaje de meses dentro del objetivo (< 15%).
- Gráfico real vs predicción con IC 95% y puntos de ventas reales.
- Tabla mes a mes con error porcentual con gradiente de color (verde < 10%, amarillo 10–15%, rojo > 15%).

**Drift alert:** si algún mes registrado supera el 15% de error, aparece un aviso recomendando reentrenar el modelo.

---

## Sección — Administración

> Disponible exclusivamente para: `admin`

Panel de gestión del sistema. Tres tabs:

**👥 Usuarios:** lista de cuentas configuradas en `secrets.toml`. Muestra nombre, rol e icono de cada usuario. No se muestran contraseñas.

**📜 Audit Log:** registro de todas las acciones críticas del sistema (login, logout, aprobación de modelo, eliminación de run). Filtrable por tipo de acción. KPIs: acciones hoy, usuarios únicos activos, alertas registradas.

**🤖 Gestión de modelos:** tabla completa de runs disponibles con métricas (MAPE, AIC, meses de datos, fecha). Permite:
- Aprobar un run: lo marca como modelo activo en producción (`activo=TRUE`).
- Eliminar un run: borra el run de la base de datos y del Storage (con confirmación).

---

## Sección — Escalabilidad

> Disponible para todos los roles

Presenta la hoja de ruta para exportar el pipeline a otras marcas, líneas de negocio y mercados de LatAm. Es la página de visión estratégica del sistema — no requiere ninguna acción del usuario, es informativa.

**Tabs disponibles:**

| Tab | Contenido |
|-----|-----------|
| 🏗️ Arquitectura | Diagrama del stack técnico y flujo de datos del sistema |
| 🚗 Portafolio | Hoja de ruta de expansión a otros modelos del portafolio Chery |
| 💼 Líneas de Negocio | Aplicación del sistema a flotillas, leasing y postventa |
| 📋 Playbook de Onboarding | Checklist para incorporar un nuevo modelo o marca en menos de 2 semanas |
| 🌎 Expansión Geográfica | Roadmap de despliegue en otros países de LatAm |
| 🚀 Visión del Producto | Evolución del sistema en 3 etapas (ver abajo) |

### Tab Visión del Producto

Muestra la evolución planificada del sistema en tres etapas:

**Etapa 1 — Sistema Reactivo (HOY, 2025-2026)** — *EN PRODUCCIÓN*
El sistema actual: predicción mensual SARIMA, dashboard de KPIs, RBAC por roles, asistente Gemini. El equipo consulta el sistema y toma decisiones de compra manualmente.

**Etapa 2 — Sistema Proactivo (AÑO 1, 2026-2027)** — *ROADMAP*
Auto-retraining mensual, soporte multi-marca en un mismo dashboard, integración con ERP del concesionario para lectura de stock en tiempo real, alertas push cuando la predicción supera umbrales configurados.

**Etapa 3 — Sistema Autónomo (AÑO 2+, 2027-2028)** — *VISIÓN ESTRATÉGICA*
Plataforma SaaS multi-tenant (varios concesionarios en una misma instancia), módulo de optimización de precio basado en elasticidad, API pública LatAm para integración con terceros.

El gráfico dual-axis muestra la evolución del valor de negocio generado (barras) y el nivel de autonomía operativa (línea) para cada etapa.

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
