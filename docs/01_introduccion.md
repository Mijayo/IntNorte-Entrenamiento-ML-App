# Introducción al Sistema de Predicción de Ventas

## ¿Qué problema resuelve?

En la comercialización de vehículos, anticipar la demanda mensual es clave para tomar decisiones de compra al fabricante, planificar inventario y asignar unidades a concesionarios. Sin embargo, la demanda de un modelo específico está sujeta a factores que varían con el tiempo: estacionalidad, tendencia de mercado, festivos, y el comportamiento de otros modelos de la misma marca.

Este sistema permite a equipos comerciales y técnicos **entrenar, evaluar y desplegar modelos de predicción de series temporales** de forma estructurada, sin necesidad de escribir código.

---

## ¿Qué hace el sistema?

El sistema cubre tres grandes áreas:

### 1. Entrenamiento de modelos SARIMA
A partir de un histórico de ventas en Excel, el sistema construye automáticamente un modelo **SARIMAX** (SARIMA con variable exógena). El proceso incluye:
- Limpieza y validación de datos
- Análisis de estacionariedad (test ADF)
- Búsqueda inteligente de hiperparámetros con **Optuna TPE**
- Validación retrospectiva con **walk-forward**
- Almacenamiento de todos los artefactos en la nube (Supabase Storage)

### 2. Dashboard de visualización
Una vez aprobado un modelo, el Dashboard muestra:
- Histórico de ventas + predicciones con intervalos de confianza al 95%
- Métricas técnicas (AIC, BIC, MAPE)
- Análisis de ventas por concesionario
- Asistente de IA conversacional (Google Gemini) adaptado al rol del usuario

### 3. Comparación Prophet vs SARIMA
Una herramienta académica que enfrenta los dos modelos más usados en predicción de series temporales de negocio: SARIMA (estadístico clásico) y Prophet (Meta, 2017), para entender cuál se adapta mejor al patrón de ventas del modelo de vehículo.

---

## Usuarios del sistema

| Rol | Descripción | Acceso |
|-----|-------------|--------|
| `admin` | Administrador técnico | Todo |
| `analyst` | Analista de datos | Entrenamiento + Dashboard completo |
| `manager` | Gerente comercial | Dashboard con recomendaciones + IA |
| `viewer` | Consultor / lectura | Solo Dashboard básico |

---

## Ejemplo de uso típico

> **Escenario**: El equipo comercial quiere saber cuántas unidades del TIGGO 2 se venderán en los próximos 6 meses para planificar la orden de compra al fabricante.

**Flujo:**
1. El analista carga el Excel con ventas históricas desde enero 2021.
2. El sistema limpia duplicados y valida que hay al menos 36 meses de datos.
3. Se lanza la búsqueda Optuna: en ~2 minutos prueba 80 combinaciones de parámetros SARIMA y selecciona la que minimiza el error porcentual (MAPE) sobre los últimos 6 meses reales.
4. Se ejecuta walk-forward: el sistema simula predecir cada mes pasado conociendo solo los datos anteriores, para medir cómo se habría desempeñado el modelo en condiciones reales.
5. El analista compara el nuevo modelo con el modelo activo en producción. Si mejora el MAPE, hace clic en **Aprobar**.
6. El Dashboard se actualiza automáticamente con las predicciones del modelo aprobado.
7. El gerente consulta el Dashboard y pregunta al asistente IA: *"¿Deberíamos comprar más o menos unidades que el mes pasado?"*

---

## Tecnologías utilizadas

| Componente | Tecnología |
|------------|-----------|
| Interfaz web | [Streamlit](https://streamlit.io) |
| Modelos estadísticos | [statsmodels](https://www.statsmodels.org) (SARIMAX) |
| Modelos de machine learning | [Prophet](https://facebook.github.io/prophet/) (Meta) |
| Optimización de hiperparámetros | [Optuna](https://optuna.org) (TPE) |
| Almacenamiento en la nube | [Supabase Storage](https://supabase.com) |
| Asistente IA | [Google Gemini](https://ai.google.dev) (gemini-2.5-flash) |
| Visualización | [Plotly](https://plotly.com/python/), [Matplotlib](https://matplotlib.org) |

---

## Estructura de la documentación

| Documento | Contenido |
|-----------|-----------|
| [01_introduccion.md](01_introduccion.md) | Este documento |
| [02_arquitectura.md](02_arquitectura.md) | Módulos, flujo de datos, Supabase Storage |
| [03_guia_usuario.md](03_guia_usuario.md) | Cómo usar el sistema paso a paso |
| [04_modelos_ml.md](04_modelos_ml.md) | SARIMA, Prophet, Optuna — conceptos y comparación |
| [05_despliegue.md](05_despliegue.md) | Configuración local y despliegue en Streamlit Cloud |
