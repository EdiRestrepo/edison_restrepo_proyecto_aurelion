# # .describe() -> Resumen completo ->  Todas las estadísticas principales
# # .info() -> Información general -> Tipos y valores nulos
# # .value_counts() -> Frecuencias -> Conteo por categoría
# # .groupby().agg() -> Estadísticas agrupadas -> Métricas por segmento

# import pandas as pd
# import seaborn as sns

# ventas = pd.read_excel("ventas.xlsx")
# clientes = pd.read_excel("clientes.xlsx")
# detalle = pd.read_excel("detalle_ventas.xlsx")
# productos = pd.read_excel("productos.xlsx")

# print(ventas.shape)
# print(ventas.describe())
# print(clientes['email'].duplicated().sum()) #ver duplicados en email

# #limpiar datos nulos
# # print(ventas.dropna(subset=['id_cliente']))

