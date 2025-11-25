# PROYECTO AURELION

# Tema:
Diseñar y documentar una BD para la gestión y consulta de información de proyecto Aurelion

# Problema:
Aurelion presenta problemas para el manejo de volúmenes altos de información de manera ordenada e integrada. No existe un sistema que permita centralizar la información en consultas rápidas y eficientes. 

# Solución:
Implementar una BD que permita integrar los cuatro datasets para mejorar la estrategia comercial en base a lo siguiente: 
- Realizar consultas.
- Centralizar consultas de clientes, productos y ventas.
- Optimizar búsquedas y generar reportes.
- Identificar patrones de consumo.
- Optiminzar inventario.

## DATASET DE REFERENCIA: Fuente, definición, estructura, tipos y escala

# FUENTE Y DEFINICIÓN
Clientes: Información personal de los clientes.
Productos: Catálogo de productos con categoría y precio unitario.
Detalle_Ventas: Detalle por producto vendido en cada venta.
Ventas: Registro de ventas con fecha, cliente y medio de pago.

# ESTRUCTURA DE DATOS
# Clientes
- id_cliente: Identificador único del cliente.
- nombre_cliente:  Nombre completo del cliente.
- email: Correo electrónico.
- ciudad: Ciudad de residencia.
- fecha_alta: Fecha de alta del cliente.

# Productos
- id_producto: Identificador único del producto.
- nombre_producto: Nombre del producto.
- categoría: clasificación del producto.
- precio_unitario: precio unitario de producto.

# Ventas
- id_venta: Identificador único de la venta.
- fecha: Fecha de la venta.
- id_cliente: Relación con Cliente.
- medio_pago: Forma de pago.

# Detalle_Ventas
- id_venta: Identificador único de la venta.
- id_producto: Identificador único del producto.
- nombre_producto: Nombre del producto.
- cantidad: Cantidad vendida.
- precio_unitario: precio unitario de producto.
- importe: importe de la venta (cantidad * precio_unitario).


# TIPOS DE DATOS y ESCALA

# 📦 Productos (productos.csv)
| Campo            | Tipo | Escala   |
|------------------|------|----------|
| id_producto      | int  | Nominal  |
| nombre_producto  | str  | Nominal  |
| categoria        | str  | Nominal  |
| precio_unitario  | int  | Razón    |

# 👥 Clientes (clientes.csv)
| Campo            | Tipo | Escala    |
|------------------|------|-----------|
| id_cliente       | int  | Nominal   |
| nombre_cliente   | str  | Nominal   |
| email            | str  | Nominal   |
| ciudad           | str  | Nominal   |
| fecha_alta       | date | Intervalo |

# 🧾 Ventas (ventas.csv)
| Campo            | Tipo | Escala    |
|------------------|------|-----------|
| id_venta         | int  | Nominal   |
| fecha            | date | Intervalo |
| id_cliente       | int  | Nominal   |
| nombre_cliente   | str  | Nominal   |
| email            | str  | Nominal   |
| medio_pago       | str  | Nominal   |

# 💰 Detalle de Ventas (detalle_ventas.csv)
| Campo             | Tipo | Escala  |
|-------------------|------|---------|
| id_venta          | int  | Nominal |
| id_producto       | int  | Nominal |
| nombre_producto   | str  | Nominal |
| cantidad          | int  | Razón   |
| precio_unitario   | int  | Razón   |
| importe           | int  | Razón   |

#Clientes puede crecer a miles, Productos a cientos y Ventas a millones.

## INFORMACIÓN, PASOS, PSEUDOCÓDIGO Y DIAGRAMA DEL PROGRAMA

La BD debe generar los siguientes reportes:
- Total de ventas por producto.
- Clientes que generan mayor gasto.
- Medios de pago más utilizados.
- Productos más vendidos.
- Ciudad que genera mayor venta.

# PASOS DEL PROGRAMA
1. Se deben cargar los 04 datasets: Clientes, Ventas, Productos, Detalle_Ventas.
2. Unir Ventas con Clientes mediante ID_Clientes.
3. Unir Detalle_Ventas con Productos mediante ID_Producto.
4. Unir Detalle_Ventas con Ventas mediante ID_Venta.
5. Calcular métricas.
6. Generar reportes y visualizaciones en power bi.

# PSEUDOCÓDIGO
Antes de empezar importar librerías:
!pip install pandas
import pandas as pd

# Paso 1: Cargar datos
clientes = pd.read_excel("clientes.xlsx")
productos = pd.read_excel("productos.xlsx")
detalle = pd.read_excel("detalle_ventas.xlsx")
ventas = pd.read_excel("ventas.xlsx")

# Paso 2: Unir datasets
ventas_clientes = ventas.merge(clientes, on="id_cliente")
detalle_productos = detalle.merge(productos, on="id_producto")
ventas_detalle = detalle_productos.merge(ventas_clientes, on="id_venta")

# Paso 3: Métricas
ventas_por_producto = ventas_detalle.groupby("nombre_producto_x")["importe"].sum()
ventas_por_cliente = ventas_detalle.groupby("nombre_cliente_x")["importe"].sum()
ventas_por_medio = ventas_detalle["medio_pago"].value_counts()
ventas_por_categoria = ventas_detalle.groupby("categoria")["importe"].sum()
ventas_por_ciudad = ventas_detalle.groupby("ciudad")["importe"].sum()

#En caso no corra, validar nombre de tablas, por ejemplo: 
print(ventas_detalle.columns.tolist())

# SUGERENCIAS Y MEJORAS APLICADAS CON COPILOT

1. Se consultaron que tipo de escala son los ids en una tabla.
2. Se consulto como programar la parte interactiva para visualizar la documentación en python por secciones.
3. Se consulto como visualizar los datos de tipo de dato y escala en una tabla en un archivo con formato .md.

## CÁLCULO DE ESTADÍSTICAS BÁSICAS

Antes de empezar importar librerías
#pip install pandas
#pip install openpyxl
import pandas as pd

# Paso 1: Cargar archivos
clientes = pd.read_excel("clientes.xlsx")
ventas = pd.read_excel("ventas.xlsx")
detalle = pd.read_excel("detalle_ventas.xlsx")
productos = pd.read_excel("productos.xlsx")

# Paso 2: Unir datasets: Se integran los datos en un solo DataFrame llamado ventas_detalle, combinando información de ventas, clientes y productos.
ventas_clientes = ventas.merge(clientes, on="id_cliente")
detalle_productos = detalle.merge(productos, on="id_producto")
ventas_detalle = detalle_productos.merge(ventas_clientes, on="id_venta")

# Paso 3: Renombrar columna para evitar ambigüedad
ventas_detalle = ventas_detalle.rename(columns={"precio_unitario_x": "precio_unitario"})

# Paso 4: Calcular estadísticas básicas
numericas = ['cantidad', 'precio_unitario', 'importe']
resumen = {}
for col in numericas:
    modo = ventas_detalle[col].mode()
    modo_val = modo.iloc[0] if not modo.empty else pd.NA
    resumen[col] = {
        'Media': ventas_detalle[col].mean(),
        'Mediana': ventas_detalle[col].median(),
        'Moda': modo_val,
        'Desviación estándar': ventas_detalle[col].std(),
        'Rango': ventas_detalle[col].max() - ventas_detalle[col].min()
    }

# Paso 5: Crear DataFrame de resumen y redondear
stats_df = pd.DataFrame(resumen)
stats_df = stats_df.round(2)

# Mostrar 'cantidad' como entero
stats_df['cantidad'] = stats_df['cantidad'].round(0).astype('Int64')

# Paso 6: Visualizar como figura: Se genera figura tipo tabla con matplotlib
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('tight')
ax.axis('off')
tabla = ax.table(cellText=stats_df.values,
                 colLabels=stats_df.columns,
                 rowLabels=stats_df.index,
                 cellLoc='center',
                 loc='center')
tabla.scale(0.8, 1.5)
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
plt.title("ESTADÍSTICAS BÁSICAS", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

# Paso 7: Métricas adicionales

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Crear carpeta para guardar figuras
output_dir = "figuras"
os.makedirs(output_dir, exist_ok=True)

# Cargar archivos
clientes = pd.read_excel("clientes.xlsx")
productos = pd.read_excel("productos.xlsx")
ventas = pd.read_excel("ventas.xlsx")
detalle = pd.read_excel("detalle_ventas.xlsx")

# Tipificar fechas y numéricos
ventas["fecha"] = pd.to_datetime(ventas["fecha"], errors="coerce")
clientes["fecha_alta"] = pd.to_datetime(clientes["fecha_alta"], errors="coerce")
for col in ["cantidad", "precio_unitario", "importe"]:
    if col in detalle.columns:
        detalle[col] = pd.to_numeric(detalle[col], errors="coerce")

# Calcular importe si falta
if "importe" not in detalle.columns or detalle["importe"].isna().any():
    detalle["importe"] = detalle["cantidad"] * detalle["precio_unitario"]

# Unificar datasets
df = detalle.merge(ventas[["id_venta", "fecha", "id_cliente", "medio_pago"]], on="id_venta", how="left")
df = df.merge(productos[["id_producto", "nombre_producto", "categoria"]], on="id_producto", how="left", suffixes=("_det", "_prod"))
df = df.merge(clientes[["id_cliente", "nombre_cliente", "ciudad"]], on="id_cliente", how="left")

# Resolver ambigüedad en nombre_producto
if "nombre_producto_det" in df.columns and "nombre_producto_prod" in df.columns:
    df["nombre_producto"] = df["nombre_producto_det"].fillna(df["nombre_producto_prod"])
elif "nombre_producto" not in df.columns:
    for candidate in ["nombre_producto_det", "nombre_producto_prod"]:
        if candidate in df.columns:
            df["nombre_producto"] = df[candidate]
            break

# Crear métricas
ventas_por_transaccion = df.groupby("id_venta", as_index=False)["importe"].sum().rename(columns={"importe": "importe_venta"})
tickets = ventas_por_transaccion["importe_venta"]

ventas_cliente = df.groupby(["id_cliente", "nombre_cliente"], as_index=False).agg(
    ingreso_total=("importe", "sum"),
    n_ventas=("id_venta", "nunique"),
    unidades=("cantidad", "sum")
).sort_values("ingreso_total", ascending=False)

ventas_producto = df.groupby(["id_producto", "nombre_producto", "categoria"], as_index=False).agg(
    ingreso_total=("importe", "sum"),
    unidades=("cantidad", "sum")
).sort_values("ingreso_total", ascending=False)
ventas_producto["precio_medio_vendido"] = ventas_producto["ingreso_total"] / ventas_producto["unidades"].replace(0, np.nan)

ventas_categoria = df.groupby("categoria", as_index=False).agg(
    ingreso_total=("importe", "sum"),
    unidades=("cantidad", "sum")
).sort_values("ingreso_total", ascending=False)

ventas_ciudad = df.groupby("ciudad", as_index=False).agg(
    ingreso_total=("importe", "sum"),
    unidades=("cantidad", "sum"),
    n_ventas=("id_venta", "nunique")
).sort_values("ingreso_total", ascending=False)

df["anio_mes"] = df["fecha"].dt.to_period("M")
ventas_mensuales = df.groupby("anio_mes", as_index=False)["importe"].sum().rename(columns={"importe": "ingreso_total"})
ventas_mensuales["anio_mes"] = ventas_mensuales["anio_mes"].astype(str)

ventas_medio_pago = df.groupby("medio_pago", as_index=False).agg(
    ingreso_total=("importe", "sum"),
    n_ventas=("id_venta", "nunique")
).sort_values("ingreso_total", ascending=False)

# Función para guardar y mostrar gráficos
def save_and_show_plot(fig, filename):
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.show()

# Gráficos
fig1, ax1 = plt.subplots(figsize=(10, 6))
top10 = ventas_producto.head(10)
ax1.barh(top10["nombre_producto"][::-1], top10["ingreso_total"][::-1])
ax1.set_title("Top 10 productos por ingreso")
ax1.set_xlabel("Ingreso total")
ax1.set_ylabel("Producto")
fig1.tight_layout()
save_and_show_plot(fig1, "top10_productos.png")

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.bar(ventas_categoria["categoria"], ventas_categoria["ingreso_total"])
ax2.set_title("Ventas por categoría")
ax2.set_xlabel("Categoría")
ax2.set_ylabel("Ingreso total")
fig2.tight_layout()
save_and_show_plot(fig2, "ventas_por_categoria.png")

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.bar(ventas_ciudad["ciudad"], ventas_ciudad["ingreso_total"])
ax3.set_title("Ventas por ciudad")
ax3.set_xlabel("Ciudad")
ax3.set_ylabel("Ingreso total")
fig3.tight_layout()
save_and_show_plot(fig3, "ventas_por_ciudad.png")

fig4, ax4 = plt.subplots(figsize=(10, 5))
ax4.plot(ventas_mensuales["anio_mes"], ventas_mensuales["ingreso_total"], marker="o")
ax4.set_title("Evolución mensual de ventas")
ax4.set_xlabel("Año-Mes")
ax4.set_ylabel("Ingreso total")
fig4.tight_layout()
save_and_show_plot(fig4, "evolucion_mensual.png")

fig5, ax5 = plt.subplots(figsize=(8, 5))
ax5.bar(ventas_medio_pago["medio_pago"], ventas_medio_pago["ingreso_total"])
ax5.set_title("Distribución de ingresos por medio de pago")
ax5.set_xlabel("Medio de pago")
ax5.set_ylabel("Ingreso total")
fig5.tight_layout()
save_and_show_plot(fig5, "medios_de_pago.png")

fig6, ax6 = plt.subplots(figsize=(6, 5))
ax6.boxplot(tickets.dropna(), vert=True, labels=["Ticket por venta"])
ax6.set_title("Distribución de ticket por venta")
ax6.set_ylabel("Importe")
fig6.tight_layout()
save_and_show_plot(fig6, "boxplot_ticket.png")

# IDENTIFICAR TIPO DE DISTRIBUCIÓN

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

# Crear carpeta de salida
output_dir = "figuras_distribucion"
os.makedirs(output_dir, exist_ok=True)

# Cargar archivos
clientes = pd.read_excel("clientes.xlsx")
productos = pd.read_excel("productos.xlsx")
ventas = pd.read_excel("ventas.xlsx")
detalle = pd.read_excel("detalle_ventas.xlsx")

# Tipificar fechas
ventas["fecha"] = pd.to_datetime(ventas["fecha"], errors="coerce")
clientes["fecha_alta"] = pd.to_datetime(clientes["fecha_alta"], errors="coerce")

# Tipificar numéricos
for col in ["cantidad", "precio_unitario", "importe"]:
    detalle[col] = pd.to_numeric(detalle[col], errors="coerce")

# Unificar datasets
ventas_cols = ["id_venta", "fecha", "id_cliente", "medio_pago"]
productos_cols = ["id_producto", "nombre_producto", "categoria"]
clientes_cols = ["id_cliente", "nombre_cliente", "ciudad"]

df = detalle.merge(ventas[ventas_cols], on="id_venta", how="left")
df = df.merge(productos[productos_cols], on="id_producto", how="left", suffixes=("_det", "_prod"))
df = df.merge(clientes[clientes_cols], on="id_cliente", how="left")

# Resolver nombre_producto
if "nombre_producto_det" in df.columns and "nombre_producto_prod" in df.columns:
    df["nombre_producto"] = df["nombre_producto_det"].fillna(df["nombre_producto_prod"])
elif "nombre_producto" not in df.columns:
    for candidate in ["nombre_producto_det", "nombre_producto_prod"]:
        if candidate in df.columns:
            df["nombre_producto"] = df[candidate]
            break

# Calcular ticket por venta
tickets = df.groupby("id_venta")["importe"].sum().rename("ticket_por_venta")
df = df.merge(tickets, on="id_venta", how="left")

# Variables numéricas a analizar
variables = ["cantidad", "precio_unitario", "importe", "ticket_por_venta"]
estadisticos = {}

# Generar gráficos y estadísticos
for var in variables:
    serie = df[var].dropna()

    # Estadísticos
    media = serie.mean()
    mediana = serie.median()
    moda = serie.mode().iloc[0] if not serie.mode().empty else np.nan
    std = serie.std()
    kurt = kurtosis(serie)
    skewness = skew(serie)
    estadisticos[var] = {
        "media": round(media, 2),
        "mediana": round(mediana, 2),
        "moda": round(moda, 2),
        "desviacion_std": round(std, 2),
        "curtosis": round(kurt, 2),
        "asimetria": round(skewness, 2)
    }

    # Histograma
    plt.figure(figsize=(8, 4))
    plt.hist(serie, bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Histograma de {var}")
    plt.xlabel(var)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/histograma_{var}.png")
    plt.show()

    # Boxplot
    plt.figure(figsize=(6, 4))
    plt.boxplot(serie, vert=True)
    plt.title(f"Boxplot de {var}")
    plt.ylabel(var)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/boxplot_{var}.png")
    plt.show()

# Convertir dict a DataFrame
estadisticos_df = pd.DataFrame(estadisticos).T
estadisticos_df.reset_index(inplace=True)
estadisticos_df.rename(columns={"index": "variable"}, inplace=True)

# Mostrar tabla
print("📊 Estadísticos por variable:")
print(estadisticos_df)

# CÁLCULO DE CORRELACIONES ENTRE VARIABLES: Se obtiene una tabla con coeficientes de correlación (de -1 a 1), con gráfico de calor que muestra visiblemente las relaciones.

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Crear carpeta de salida
output_dir = "figuras_correlacion"
os.makedirs(output_dir, exist_ok=True)

# Cargar archivos
clientes = pd.read_excel("clientes.xlsx")
productos = pd.read_excel("productos.xlsx")
ventas = pd.read_excel("ventas.xlsx")
detalle = pd.read_excel("detalle_ventas.xlsx")

# Tipificar fechas y numéricos
ventas["fecha"] = pd.to_datetime(ventas["fecha"], errors="coerce")
for col in ["cantidad", "precio_unitario", "importe"]:
    detalle[col] = pd.to_numeric(detalle[col], errors="coerce")

# Calcular importe si falta
if "importe" not in detalle.columns or detalle["importe"].isna().any():
    detalle["importe"] = detalle["cantidad"] * detalle["precio_unitario"]

# Unificar datasets
df = detalle.merge(ventas[["id_venta", "fecha", "id_cliente", "medio_pago"]], on="id_venta", how="left")
df = df.merge(productos[["id_producto", "nombre_producto", "categoria"]], on="id_producto", how="left")
df = df.merge(clientes[["id_cliente", "nombre_cliente", "ciudad"]], on="id_cliente", how="left")

# Calcular ticket por venta
tickets = df.groupby("id_venta")["importe"].sum().rename("ticket_por_venta")
df = df.merge(tickets, on="id_venta", how="left")

# Seleccionar variables numéricas
variables = ["cantidad", "precio_unitario", "importe", "ticket_por_venta"]
df_numerico = df[variables].dropna()

# Calcular matriz de correlación
correlaciones = df_numerico.corr()

# Mostrar tabla
print("📊 Matriz de correlación:")
print(correlaciones.round(2))

# Visualizar como heatmap y guardar como PNG
plt.figure(figsize=(8, 6))
sns.heatmap(correlaciones, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("🔗 Matriz de correlación entre variables")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlacion_heatmap.png"), dpi=150)
plt.show()

# ANÁLISIS DE OUTLIERS:  Genera una tabla resumen con total de registros, número y porcentaje de outliers, valores mínimos, máximos, Q1, Q3 e IQR. Dado que ya contamos con los gráficos en la sección Distribución, este muestra el resumen

import matplotlib.pyplot as plt
import os
import pandas as pd

# Asegurar carpeta de salida
output_dir = "figuras_outliers"
os.makedirs(output_dir, exist_ok=True)

# Variables numéricas que ya analizaste
variables = ["cantidad", "precio_unitario", "importe", "ticket_por_venta"]
outlier_summary = {}

# Calcular resumen de outliers por variable
for var in variables:
    serie = df[var].dropna()
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = serie[(serie < lower) | (serie > upper)]

    outlier_summary[var] = {
        "Total registros": len(serie),
        "Outliers": len(outliers),
        "% Outliers": round(len(outliers) / len(serie) * 100, 2),
        "Min": round(serie.min(), 2),
        "Max": round(serie.max(), 2),
        "Q1": round(Q1, 2),
        "Q3": round(Q3, 2),
        "IQR": round(IQR, 2)
    }

# Convertir a DataFrame
outliers_df = pd.DataFrame(outlier_summary).T.reset_index().rename(columns={"index": "Variable"})

# Mostrar como figura PNG
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')
tabla = ax.table(cellText=outliers_df.values,
                 colLabels=outliers_df.columns,
                 cellLoc='center',
                 loc='center')
tabla.scale(1.2, 1.5)
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
plt.title("📊 Resumen de Outliers por Variable", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "tabla_outliers_resumen.png"), dpi=150)
plt.show()

# INTEPRETACIÓN DE RESULTADOS
1.  Centralización de la información
Se integraron los 04 datasets, lo que permitió una visión única de cada transacción: quién compró, qué compró, cuánto pagó, cómo pagó, lugar de compra.
Esto resuelve el problema inicial sobre datos dispersos y no relacionados, lo cual impedía consultas rápidas.
2. Patrones de Consumo y ventas
Se identificaron productos más vendidos y categorías con mayor ingreso, lo cual ayuda a priorizar el inventario y campañas de marketing en los productos más rentables.
3. Top 10 de productos por ingreso
Gráficos muestran qué productos generan la mayor parte de las ventas, lo cual permite aplicar la regla de pareto (80/20): concentrarse en el 20% de productos que generan más ingresos.
4. Clientes más valiosos
El reporte de ventas por cliente muestra quiénes generan mayor gasto, lo cual permite que se puedan segmentar los clientes VIP y diseñar estrategias de fidelización.
También se detecta el volumen de ventas por ciudad, lo que ayuda a identificar mercados más rentables.
5. Medios de pago más utilizados
El análisis de ventas por medio de pago revela las preferencias de los clientes, lo cual permite optimizar acuerdos con bancos, pasarelas de pago y reducir costos de transacción.
6. Evolución temporal
El gráfico de ventas mensuales muestra tendencias y estacionalidad, lo cual ayuda a planificar inventaraio y campañas en los meses que hay mayor demanda.
Además, el análisis de tickets por venta (boxplot) muestra la dispersión del gasto por transacción, útil para definir estrategias de upselling.
7. Estadísticas y distribución
Se calcularon media, mediana, moda, desviación estándar, lo que nos permite entender si las ventas están concentradas en pocos valores (asimetría positiva) o si hay dispersión amplia (alta desviación estándar).
8. Correlaciones
La matriz de correlación muestra relaciones entre variables, como por ejemplo: cantidad vs importe (alta correlación) confirma que más unidades vendidas generan mayor ingreso.
También se valida Precio unitario vs ticket por venta, lo cual puede mostrar si los productos más caros realmente elevan el gasto por transacción.
9. Outliers
El análisis de outliers detecta transacciones atípicas (ej. compras extremadamente altas o ba   jas).
Esto nos ayuda a identificar errores de carga de datos, detectar clientes con comportamientos especiales, ajustar estrategias de inventario y precios.

# CONCLUSIONES
- Ventas con variaciones temporales significativas (Meses Abril y Mayo).
- Concentración de ventas en pocas ciudades (Rio Cuarto-Altagracia- Córdova).
- Preferencia marcada por ciertos medios de pago (efectivo).
- No se detecta correlación fuerte entre cantidad y precio unitario.
- El sistema diseñado responde directamente al problema de Aurelion: Ordena y centraliza la información dispersa.
- El sistema optimiza consultas y genera reportes clave para la toma de decisiones.
- También identifica patrones de consumo y clientes estratégicos.
- El sistema mejora la gestión de inventario al saber qué productos y categorías tienen mayor rotación de productos.
- Finalmente, apoya la estrategia comercial con insights sobre ciudades más rentables y medios de pago preferidos.