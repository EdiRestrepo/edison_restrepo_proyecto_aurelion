# Instalar librerías si es necesario
# !pip install pandas matplotlib openpyxl
import pandas as pd
import matplotlib.pyplot as plt

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

#IDENTIFICAR TIPO DE DISTRIBUCIÓN 

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


# CÁLCULO DE CORRELACIONES ENTRE VARIABLES PRINCIPALES

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

#ANÁLISIS DE OUTLIERS

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