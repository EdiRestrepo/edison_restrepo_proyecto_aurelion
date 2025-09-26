
# Tema
Integración y análisis de datos de ventas y clientes usando Excel y Python en Visual Studio Code.

# Problema
Se requiere procesar y analizar datos de ventas, productos y clientes almacenados en archivos Excel para generar reportes y métricas. El entorno de desarrollo debe ser reproducible y alineado con las configuraciones del proyecto.

# Solución
- Usar Python para el procesamiento y análisis (pandas, openpyxl/xlrd según versión).
- Desarrollar scripts y notebooks en Visual Studio Code con las extensiones de Python y Jupyter.
- Mantener la configuración de editor en la carpeta [.vscode](.vscode/settings.json) y símbolos adicionales en [.vscode/excel-pq-symbols/excel-pq-symbols.json](.vscode/excel-pq-symbols/excel-pq-symbols.json).
- Los archivos Excel base están en el repositorio local: [clientes.xlsx](clientes.xlsx), [detalle_ventas.xlsx](detalle_ventas.xlsx), [productos.xlsx](productos.xlsx), [ventas.xlsx](ventas.xlsx). La carpeta `.vscode` y los Excel originales provienen de la URL del drive: https://drive.google.com/drive/folders/1EHGn5ZIYNI5pXE53pbXxGiIDt-ecnPU8

-------------------------------------

# Estructura
- Raíz del proyecto:
  - [documentacion.md](documentacion.md)
  - [clientes.xlsx](clientes.xlsx)
  - [detalle_ventas.xlsx](detalle_ventas.xlsx)
  - [productos.xlsx](productos.xlsx)
  - [ventas.xlsx](ventas.xlsx)
  - .vscode/
    - [.vscode/settings.json](.vscode/settings.json)
    - [.vscode/excel-pq-symbols/excel-pq-symbols.json](.vscode/excel-pq-symbols/excel-pq-symbols.json)

# Tipos
- Archivos de datos: Excel (.xlsx)
- Código: Python (.py) y Notebooks (.ipynb)
- Configuración del editor: JSON (en `.vscode`)

# Escalado de la BD
- Para volúmenes mayores: migrar a un almacén columnar (parquet) o base de datos relacional (Postgres).
- Pipeline recomendado: ingestión -> limpieza (pandas) -> particionado por fecha/producto -> almacenamiento en parquet o BD.
- Automatización: usar scripts programados (cron / task scheduler) o un job en Airflow/Prefect para procesos recurrentes.

--------------------------------------

# Información
- Símbolos de Power Query incluidos en el proyecto:
  - [`Excel.CurrentWorkbook`](.vscode/excel-pq-symbols/excel-pq-symbols.json)
  - [`Documentation`](.vscode/excel-pq-symbols/excel-pq-symbols.json)
  (ver [.vscode/excel-pq-symbols/excel-pq-symbols.json](.vscode/excel-pq-symbols/excel-pq-symbols.json) para detalles)

# Pasos (instalación y configuración)
1. Instalar Python (recomendado: 3.10+).  
2. Instalar Visual Studio Code.  
3. Abrir el proyecto en VS Code.  
4. Instalar extensiones:
   - Python (ms-python.python)
   - Jupyter (ms-toolsai.jupyter)
5. Confirmar que la carpeta `.vscode` está presente y que [settings.json](.vscode/settings.json) apunta a los símbolos extras (opcional editar ruta).
6. Descargar la carpeta y archivos Excel desde: https://drive.google.com/drive/folders/1EHGn5ZIYNI5pXE53pbXxGiIDt-ecnPU8 y colocarlos en la raíz del proyecto si aún no están.
7. Crear y activar un entorno virtual:
   - python -m venv .venv
   - Windows: .venv\Scripts\activate
   - Unix: source .venv/bin/activate
8. Instalar dependencias:
   - pip install pandas openpyxl jupyterlab
9. Abrir o crear notebooks (.ipynb) o scripts (.py) para análisis en VS Code usando la extensión Jupyter.

# pseudocódigo
- Cargar datos:
  - clientes = pd.read_excel('clientes.xlsx')
  - productos = pd.read_excel('productos.xlsx')
  - ventas = pd.read_excel('ventas.xlsx')
  - detalle = pd.read_excel('detalle_ventas.xlsx')
- Limpiar y validar:
  - normalizar columnas, convertir fechas, eliminar duplicados
- Unir tablas:
  - ventas_detalle = detalle.merge(ventas, on='venta_id').merge(productos, on='producto_id')
  - ventas_completas = ventas_detalle.merge(clientes, on='cliente_id')
- Agregar métricas:
  - total_por_producto = ventas_completas.groupby('producto_id')['importe'].sum()
  - top_clientes = ventas_completas.groupby('cliente_id')['importe'].sum().nlargest(10)
- Exportar resultados:
  - total_por_producto.to_excel('reportes/total_por_producto.xlsx')

