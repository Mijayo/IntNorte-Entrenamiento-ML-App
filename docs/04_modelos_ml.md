# Modelos de Machine Learning

Este documento explica los conceptos clave detrás de los modelos usados en el sistema: **SARIMA**, **Prophet** y **Optuna TPE**. El objetivo es que cualquier persona con formación básica en estadística o negocio pueda entender qué hace cada modelo, por qué se eligió y cuáles son sus limitaciones.

---

## 1. Series temporales: el problema que resolvemos

Una **serie temporal** es una secuencia de observaciones ordenadas en el tiempo. En nuestro caso, cada observación es el número de vehículos vendidos en un mes determinado.

```
Mes        Ventas
2024-01      38
2024-02      41
2024-03      35
2024-04      52
...
```

El objetivo de los modelos es aprender los **patrones históricos** (tendencia, estacionalidad, ciclos) para **extrapolarlos al futuro**.

### Conceptos básicos

**Tendencia:** Dirección general de la serie a largo plazo. Si las ventas crecen año tras año, la serie tiene tendencia positiva.

**Estacionalidad:** Patrones que se repiten con una frecuencia fija. En ventas de coches es habitual ver picos en determinados meses del año (por campañas, fin de ejercicio fiscal, etc.).

**Estacionariedad:** Una serie es estacionaria cuando su media y varianza no cambian con el tiempo. La mayoría de modelos ARIMA requieren que la serie sea (o se transforme para ser) estacionaria.

---

## 2. Modelo SARIMA

### ¿Qué es?

**SARIMA** son las siglas de *Seasonal AutoRegressive Integrated Moving Average*. Es el modelo estadístico clásico para series temporales con estacionalidad.

Se escribe como **SARIMA(p, d, q)(P, D, Q, m)** donde cada letra es un hiperparámetro:

| Parámetro | Nombre | ¿Qué controla? |
|-----------|--------|----------------|
| `p` | AR (AutoRegresivo) | Cuántos meses pasados influyen directamente en el valor actual |
| `d` | Integración | Cuántas veces hay que diferenciar la serie para hacerla estacionaria |
| `q` | MA (Media Móvil) | Cuántos errores pasados del modelo influyen en el valor actual |
| `P` | AR estacional | Igual que `p`, pero a escala anual |
| `D` | Integración estacional | Igual que `d`, pero a escala anual |
| `Q` | MA estacional | Igual que `q`, pero a escala anual |
| `m` | Período estacional | Número de períodos en un ciclo (12 para datos mensuales) |

**Ejemplo:** SARIMA(1,1,1)(1,1,1,12) significa:
- `p=1`: el valor de este mes depende del mes anterior
- `d=1`: se diferencia una vez para eliminar la tendencia
- `q=1`: el error del mes anterior también se incorpora al modelo
- Los mismos principios se aplican a escala anual con `P=D=Q=1`

### Variable exógena: SARIMAX

En este sistema se usa la variante **SARIMAX** (la X es de e**X**ógena), que añade una variable externa como señal adicional. La variable exógena es el **total mensual de ventas de los demás modelos de la misma marca**.

La intuición es que si en un mes se venden muchas unidades de otros modelos CHERY, es probable que el contexto de mercado (campañas, eventos) también favorezca las ventas del TIGGO 2. Esta señal ayuda al modelo a capturar efectos que la propia serie del TIGGO 2 no tiene suficiente historia para detectar.

### Estacionariedad y test ADF

Antes de entrenar, el sistema verifica si la serie es estacionaria con el **test de Dickey-Fuller Aumentado (ADF)**:

- **Hipótesis nula (H₀):** la serie tiene raíz unitaria (no es estacionaria)
- Si el p-valor < 0.05, se rechaza H₀ → la serie es estacionaria
- Si el p-valor ≥ 0.05, la serie necesita diferenciación (`d > 0`)

```
Test ADF para serie TIGGO 2 (2021-01 a 2026-02):
  Estadístico: -2.31
  p-valor: 0.0178  → ✅ Serie estacionaria (p < 0.05)
```

### Gráficos ACF y PACF

Para elegir buenos valores iniciales de `p` y `q`, se usan dos gráficos:

- **ACF (Autocorrelation Function):** muestra cuánto se correlaciona la serie consigo misma a distintos desplazamientos (lags). Un pico significativo en lag=k indica que el valor de hace k meses sigue siendo relevante.
- **PACF (Partial Autocorrelation Function):** similar al ACF pero elimina las correlaciones intermedias. Ayuda a determinar el valor de `p`.

En la práctica, la búsqueda automática con Optuna hace que no sea necesario interpretarlos para elegir parámetros, pero siguen siendo útiles para entender la estructura de la serie.

---

## 3. Optuna TPE — Búsqueda inteligente de hiperparámetros

### ¿Por qué no un grid search exhaustivo?

El espacio de posibles combinaciones de parámetros SARIMA es grande. Con `p ∈ {0,1,2,3}`, `d ∈ {0,1}`, `q ∈ {0,1,2,3}`, `P ∈ {0,1}`, `D ∈ {0,1}`, `Q ∈ {0,1,2}` hay **384 combinaciones posibles**. Cada combinación requiere entrenar y evaluar un modelo completo, lo que puede tardar varios minutos en total.

Además, el grid search es **ciego**: evalúa todas las combinaciones con la misma prioridad, sin aprender de los intentos anteriores.

### ¿Qué es Optuna TPE?

[Optuna](https://optuna.org) es una librería de optimización de hiperparámetros que implementa el algoritmo **TPE (Tree-structured Parzen Estimator)**. Es un método de **optimización bayesiana**: usa los resultados de los trials anteriores para inferir qué regiones del espacio tienen más probabilidad de contener el mínimo.

En términos simples: si los 20 primeros intentos con `p=1` dan mejores resultados que con `p=3`, Optuna muestreará más combinaciones con `p=1` en los intentos siguientes.

**Comparación:**

| Método | Combinaciones evaluadas | Uso de resultados anteriores | Tiempo aprox. |
|--------|:-----------------------:|:----------------------------:|:-------------:|
| Grid search exhaustivo | 384 (todas) | No | ~8 min |
| Optuna TPE | 80 (configurado) | Sí (bayesiano) | ~2 min |

Con 80 trials, Optuna típicamente encuentra un modelo con MAPE igual o mejor que el grid search completo, en ~4× menos tiempo.

### Criterio de selección

El criterio de optimización es **minimizar el MAPE** (error porcentual medio) sobre el conjunto de test. No se usa el AIC como criterio principal porque el AIC mide el ajuste sobre los datos de entrenamiento (in-sample), mientras que el MAPE sobre el test mide la capacidad predictiva real (out-of-sample), que es lo que nos interesa en producción.

### Trials descartados

Un trial se descarta (y no cuenta como válido) si ocurre cualquiera de estas condiciones:

1. **Predicciones negativas:** el modelo predice ventas < 0, lo que es imposible en la realidad.
2. **Predicciones fuera de rango:** el modelo predice ventas > `max_ventas` configurado.
3. **Error numérico:** el modelo no converge o produce valores infinitos o NaN.

Los trials descartados **son normales** y esperables — indican que esa combinación de parámetros no es adecuada para esta serie. Un porcentaje de descarte del 15–30% es completamente normal.

---

## 4. Walk-Forward Validation

### ¿Qué es?

La validación walk-forward (también llamada *rolling origin* o *backtesting*) es el método más riguroso para evaluar un modelo de predicción temporal.

A diferencia de una división simple train/test, el walk-forward simula exactamente lo que ocurriría si hubiéramos usado el modelo en producción: cada predicción se hace conociendo solo los datos disponibles hasta ese momento.

**Ejemplo con 6 meses de validación:**

```
Mes  │ Datos usados para entrenar    │ Se predice  │ Real
─────┼────────────────────────────────┼─────────────┼──────
Sep  │ ene 2021 ... ago 2025          │ Sep 2025    │  44
Oct  │ ene 2021 ... sep 2025          │ Oct 2025    │  38
Nov  │ ene 2021 ... oct 2025          │ Nov 2025    │  51
Dic  │ ene 2021 ... nov 2025          │ Dic 2025    │  39
Ene  │ ene 2021 ... dic 2025          │ Ene 2026    │  42
Feb  │ ene 2021 ... ene 2026          │ Feb 2026    │  47
```

El MAPE walk-forward es la media de los errores porcentuales de esas 6 predicciones individuales. Es el indicador más confiable de cómo se comportará el modelo en producción.

### MAPE: Error Porcentual Medio

$$\text{MAPE} = \frac{1}{n} \sum_{t=1}^{n} \left| \frac{y_t - \hat{y}_t}{y_t} \right| \times 100$$

Donde `y_t` es el valor real y `ŷ_t` es el valor predicho.

**Interpretación orientativa:**

| MAPE | Calidad del modelo |
|------|-------------------|
| < 5% | Excelente |
| 5–10% | Bueno |
| 10–20% | Aceptable |
| > 20% | Mejorable |

> En series de ventas de vehículos con alta variabilidad (festivos, lanzamientos, escasez de stock), un MAPE del 8–12% suele ser un resultado muy aceptable.

---

## 5. Modelo Prophet

### ¿Qué es?

**Prophet** es un modelo de predicción de series temporales desarrollado por Meta (Facebook) y publicado en 2017. Está diseñado para series con **estacionalidad múltiple**, **tendencia no lineal** y **efectos de días especiales (festivos)**.

A diferencia de SARIMA, Prophet no es un modelo ARIMA. Su ecuación es aditiva:

```
y(t) = tendencia(t) + estacionalidad(t) + festivos(t) + ruido(t)
```

Cada componente se modela por separado:
- **Tendencia:** curva logística o lineal con *changepoints* automáticos.
- **Estacionalidad:** series de Fourier para capturar patrones anuales y semanales.
- **Festivos:** efecto de cada día festivo, modelado como un impacto aditivo.

### Configuración usada en este sistema

```python
Prophet(
    yearly_seasonality=True,     # Estacionalidad anual
    weekly_seasonality=False,    # Desactivada (datos mensuales)
    daily_seasonality=False,     # Desactivada (datos mensuales)
    seasonality_mode='multiplicative'  # Adecuado cuando la amplitud
                                       # estacional crece con la tendencia
)
m.add_country_holidays(country_name='MX')  # Festivos mexicanos
```

La **estacionalidad multiplicativa** es apropiada cuando los picos y valles estacionales son proporcionalmente más grandes en períodos de mayor demanda (por ejemplo, si diciembre es siempre un 30% mejor que la media, no un +10 unidades fijo).

---

## 6. SARIMA vs Prophet — ¿cuándo gana cada uno?

| Situación | Ventaja |
|-----------|---------|
| Serie larga y estable (>5 años, sin quiebres) | SARIMA |
| Tendencia cambiante o quiebres estructurales | Prophet |
| Festivos con impacto medible | Prophet |
| Variable exógena relevante disponible | SARIMA (SARIMAX) |
| Datos faltantes o irregulares | Prophet |
| Interpretación de componentes | Prophet (descomposición visual) |
| Calibración automática eficiente | Ambos (Prophet requiere menos trials) |
| Series cortas (< 3 años) | Prophet suele ser más robusto |

**Regla práctica:** si SARIMA y Prophet dan MAPEs similares (diferencia < 1%), conviene usar **Prophet en producción** por su menor necesidad de ajuste manual y su mejor manejo de festivos. Si SARIMA supera a Prophet claramente, probablemente la variable exógena (`ventas_otros`) está aportando información que Prophet no puede capturar.

---

## 7. Intervalos de confianza

El modelo final genera predicciones puntuales (el valor más probable) y un **intervalo de confianza al 95%** para cada mes futuro.

```
Mes        Predicción   IC Inferior   IC Superior
Mar 2026       47.2         38.1          56.3
Abr 2026       44.8         34.2          55.4
May 2026       51.3         38.9          63.7
```

El intervalo indica que, con un 95% de probabilidad, las ventas reales caerán dentro de ese rango. El intervalo se **ensancha** conforme nos alejamos en el tiempo, reflejando la mayor incertidumbre de predicciones lejanas.

**Uso recomendado:**
- Para decisiones de compra conservadoras: usar el **IC inferior**.
- Para planificación de capacidad máxima: usar el **IC superior**.
- Para el punto de referencia central: usar la **predicción puntual**.
