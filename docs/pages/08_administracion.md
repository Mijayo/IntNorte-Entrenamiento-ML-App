# Página 8 — Panel de Administración

**Archivo:** `pages/8_Administracion.py`  
**Acceso:** exclusivamente `admin`  
**Icono:** ⚙️

---

## Propósito

Panel de gestión del sistema. Centraliza la administración de usuarios, el audit log de actividad y la gestión de modelos entrenados (activación y eliminación de runs).

---

## Tab 👥 Usuarios

Lista de cuentas configuradas en `.streamlit/secrets.toml`. **Las contraseñas nunca se muestran.**

**KPIs:**

| KPI | Descripción |
|-----|-------------|
| Total usuarios | Número de cuentas configuradas |
| Con Supabase Auth | Cuentas con email (autenticación vía Supabase) |
| Admins | Cuentas con rol `admin` |

**Tabla de usuarios:**

| Columna | Descripción |
|---------|-------------|
| Usuario | Nombre de usuario |
| Nombre | Nombre completo (campo `name` en secrets) |
| Rol | Icono + etiqueta del rol |
| Email | Email de la cuenta |
| Auth Supabase | ✅ si tiene email / 🔑 Local si no |
| Permisos | Lista de permisos activos (`True`) del campo `permissions` |

**Iconos de rol:**

| Rol | Icono |
|-----|-------|
| admin | 👑 Admin |
| manager | 💼 Gerente |
| analyst | 📊 Analista |
| financiero | 💰 Financiero |
| viewer | 👁 Viewer |

**Para añadir, modificar o eliminar usuarios:** edita `.streamlit/secrets.toml` y haz redeploy de la aplicación en Streamlit Cloud.

---

## Tab 📜 Audit Log

Registro de todas las acciones críticas del sistema almacenadas en Supabase.

**Slider** para seleccionar cuántas entradas mostrar (10–500, por defecto 50).

**KPIs:**

| KPI | Descripción |
|-----|-------------|
| Total acciones | Número de entradas en el período cargado |
| Logins | Acciones de tipo `LOGIN` |
| Entrenamientos | Acciones de tipo `SAVE_TRAINING` |
| Aprobaciones | Acciones de tipo `APPROVE_MODEL` |

**Filtro por acción** — selectbox con todas las acciones distintas presentes en el log.

**Columnas de la tabla:** timestamp · usuario · accion · run_name · detalle.

**Gráfico de actividad diaria** — barras del número de acciones por día.

---

## Tab 🤖 Gestión de modelos

Gestión completa del ciclo de vida de los runs entrenados.

**KPIs:**

| KPI | Descripción |
|-----|-------------|
| Modelos disponibles | Total de runs en Supabase |
| Modelo activo | Label del run activo actualmente en producción |

### Tabla de runs

Columnas: run_name · created_at · usuario · mape_wf · aic · activo (🟢 Activo / —).

### Cambiar modelo activo

- Selectbox con los runs no activos.
- Botón **✅ Activar este modelo** — llama a `sio.approve_model()`, que marca el run seleccionado como `activo=TRUE` en Supabase y desactiva el anterior.

### Eliminar modelo

> ⚠️ Esta acción es **irreversible**. Los artefactos se borran tanto de Supabase Storage como de la base de datos.

- No se puede eliminar el modelo activo — debes activar otro primero.
- Checkbox de confirmación obligatorio antes de poder pulsar el botón de eliminación.
- Botón **🗑️ Eliminar modelo seleccionado** (estilo `secondary`) — deshabilitado hasta confirmar.

---

## Permisos y restricciones

- La página aplica `guard_page(roles=["admin"])` — cualquier usuario con rol distinto de `admin` ve una pantalla de acceso restringido.
- Las operaciones de activación y eliminación se registran automáticamente en el audit log.
