# Documentación por Página — Sistema TIGGO 2

Cada archivo documenta una página de la aplicación Streamlit: propósito, acceso por rol, tabs, parámetros, funciones y flujo de uso.

| Página | Archivo | Acceso | Descripción |
|--------|---------|--------|-------------|
| [1 — Entrenamiento](01_entrenamiento.md) | `pages/1_Entrenamiento.py` | admin, analyst | Carga de datos, validación, entrenamiento SARIMA con Optuna TPE y gestión del ciclo de vida del modelo |
| [2 — Comparativa ML](02_comparativa_ml.md) | `pages/2_Comparativa_ML.py` | admin, analyst | Comparativa de 5 modelos (SARIMAX, Prophet, LR, RF, XGBoost) sobre el mismo histórico |
| [3 — Dashboard](03_dashboard.md) | `pages/3_Dashboard.py` | todos los roles | Dashboard principal con KPIs, predicciones, recomendaciones, métricas técnicas y asistente IA |
| [4 — Concesionarios](04_concesionarios.md) | `pages/4_Concesionarios.py` | admin, analyst, manager | Análisis histórico y predicciones desagregadas por tienda usando shares históricos |
| [5 — Proyección Ingresos](05_proyeccion_ingresos.md) | `pages/5_Proyeccion_Ingresos.py` | admin, analyst, financiero | Proyección financiera a 6 meses en USD + calculadora de ROI estratégico |
| [6 — Escalabilidad](06_escalabilidad.md) | `pages/6_Escalabilidad.py` | todos los roles | Hoja de ruta multi-marca, portafolio de expansión, líneas de negocio y expansión LatAm |
| [7 — Registrar Ventas](07_registrar_ventas.md) | `pages/7_Registrar_Ventas.py` | admin, analyst | Feedback loop: ingreso de ventas reales y comparativa forecast vs producción |
| [8 — Administración](08_administracion.md) | `pages/8_Administracion.py` | admin | Gestión de usuarios, audit log y ciclo de vida de modelos entrenados |
