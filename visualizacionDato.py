# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Cargar los datos
# clientes = pd.read_excel("clientes.xlsx")
# productos = pd.read_excel("productos.xlsx")
# ventas = pd.read_excel("ventas.xlsx")
# detalle = pd.read_excel("detalle_ventas.xlsx")

# # Unir las tablas
# df = detalle.merge(ventas, on="id_venta")
# df = df.merge(productos, on="id_producto")
# df = df.merge(clientes, on="id_cliente")

# # Mostrar columnas disponibles
# print(df.columns)

# # Convertir a fecha y crear columna mes
# df['fecha'] = pd.to_datetime(df['fecha'])
# df['mes'] = df['fecha'].dt.to_period('M')

# # Calcular ventas mensuales
# ventas_mensuales = df.groupby('mes')['total'].sum().reset_index()

# # Graficar evolución
# plt.figure(figsize=(10, 5))
# sns.lineplot(data=ventas_mensuales, x='mes', y='total', marker='o')
# plt.title("Evolución de Ventas Mensuales")
# plt.xlabel("Mes")
# plt.ylabel("Ventas Totales")
# plt.grid(True)
# plt.show()
