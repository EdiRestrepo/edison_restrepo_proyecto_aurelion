# # pandas-estadistica.py
# # -*- coding: utf-8 -*-
# """
# Script de análisis de datos con pandas y matplotlib
# ---------------------------------------------------
# Realiza:
# 1. Carga automática de archivos .xlsx
# 2. Limpieza y unificación de los datos
# 3. Cálculo de totales, promedios y desviaciones
# 4. Análisis por cliente, producto, categoría y ciudad
# 5. Evolución temporal mensual
# 6. Distribución de medios de pago
# 7. Gráficos con matplotlib
# """

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # -------------------------------------------------------------------
# # 1️⃣ CARGA AUTOMÁTICA DE ARCHIVOS .XLSX
# # -------------------------------------------------------------------
# print("Cargando archivos...")

# clientes = pd.read_excel("clientes.xlsx")
# productos = pd.read_excel("productos.xlsx")
# ventas = pd.read_excel("ventas.xlsx")
# detalle = pd.read_excel("detalle_ventas.xlsx")

# print("Archivos cargados correctamente.")

# # -------------------------------------------------------------------
# # 2️⃣ LIMPIEZA Y UNIFICACIÓN DE DATOS
# # -------------------------------------------------------------------
# print("Limpiando y unificando datos...")

# ventas["fecha"] = pd.to_datetime(ventas["fecha"], errors="coerce")
# clientes["fecha_alta"] = pd.to_datetime(clientes["fecha_alta"], errors="coerce")

# # Convertir columnas numéricas
# for col in ["cantidad", "precio_unitario", "importe"]:
#     if col in detalle.columns:
#         detalle[col] = pd.to_numeric(detalle[col], errors="coerce")

# # Calcular importe si falta
# if "importe" not in detalle.columns or detalle["importe"].isna().any():
#     detalle["importe"] = detalle["cantidad"] * detalle["precio_unitario"]

# # Unificar datasets (detalle + ventas + productos + clientes)
# df = (
#     detalle.merge(ventas[["id_venta", "fecha", "id_cliente", "medio_pago"]], on="id_venta", how="left")
#     .merge(productos[["id_producto", "nombre_producto", "categoria"]], on="id_producto", how="left")
#     .merge(clientes[["id_cliente", "nombre_cliente", "ciudad"]], on="id_cliente", how="left")
# )

# print("Datos unificados correctamente.")

# # --- VALIDACIONES Y NORMALIZACIÓN DE COLUMNAS (nuevo) ---
# # A veces los archivos excel vienen con nombres distintos; intentar normalizar
# expected_cols = {"id_producto","nombre_producto","categoria","cantidad","precio_unitario","importe","id_venta","id_cliente","nombre_cliente","ciudad","fecha","medio_pago"}
# present = set(df.columns)

# if not expected_cols.issubset(present):
#     missing = expected_cols - present
#     print(f"Advertencia: faltan columnas esperadas en df: {missing}. Intentando mapear alternativas conocidas...")

#     # Mapear nombres alternativos comunes a los esperados
#     alt_map = {}
#     if "product_name" in df.columns: alt_map["product_name"] = "nombre_producto"
#     if "nombre producto" in df.columns: alt_map["nombre producto"] = "nombre_producto"
#     if "categoria_producto" in df.columns: alt_map["categoria_producto"] = "categoria"
#     if "qty" in df.columns: alt_map["qty"] = "cantidad"
#     if "price" in df.columns: alt_map["price"] = "precio_unitario"
#     if "amount" in df.columns: alt_map["amount"] = "importe"
#     if "customer_name" in df.columns: alt_map["customer_name"] = "nombre_cliente"
#     if "city" in df.columns: alt_map["city"] = "ciudad"
#     if "payment_method" in df.columns: alt_map["payment_method"] = "medio_pago"

#     if alt_map:
#         df = df.rename(columns=alt_map)
#         print("Renombradas columnas:", alt_map)

#     # Recalcular conjunto presente y faltantes
#     present = set(df.columns)
#     missing = expected_cols - present
#     if missing:
#         print("Columnas que siguen faltando (se rellenarán con valores por defecto si es necesario):", missing)

# # Asegurar columnas críticas tienen alguna columna usable
# if "nombre_producto" not in df.columns:
#     df["nombre_producto"] = "Desconocido"
# if "categoria" not in df.columns:
#     df["categoria"] = "Sin categoría"
# if "cantidad" not in df.columns:
#     df["cantidad"] = 0
# if "precio_unitario" not in df.columns:
#     df["precio_unitario"] = 0.0
# if "importe" not in df.columns:
#     # ya hay lógica previa para calcular importe, pero asegurar por si acaso
#     df["importe"] = df["cantidad"].fillna(0) * df["precio_unitario"].fillna(0)

# # Convertir tipos numéricos seguros (repetir conversión por si renombramos columnas)
# for col in ["cantidad", "precio_unitario", "importe"]:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# # -------------------------------------------------------------------
# # 3️⃣ CÁLCULO DE TOTALES, PROMEDIOS Y DESVIACIONES
# # -------------------------------------------------------------------
# print("Calculando métricas generales...")

# ventas_por_venta = df.groupby("id_venta", as_index=False)["importe"].sum().rename(columns={"importe": "importe_venta"})
# tickets = ventas_por_venta["importe_venta"]

# resumen = {
#     "Periodo Inicio": [df["fecha"].min().date()],
#     "Periodo Fin": [df["fecha"].max().date()],
#     "N° Ventas": [ventas_por_venta["id_venta"].nunique()],
#     "Líneas Detalle": [len(df)],
#     "Unidades Vendidas": [int(df["cantidad"].sum())],
#     "Ingreso Total": [float(df["importe"].sum())],
#     "Ticket Promedio (Venta)": [tickets.mean()],
#     "Desviación Ticket": [tickets.std()],
# }
# resumen_df = pd.DataFrame(resumen).round(2)
# print(resumen_df)

# # -------------------------------------------------------------------
# # 4️⃣ ANÁLISIS POR CLIENTE, PRODUCTO, CATEGORÍA Y CIUDAD
# # -------------------------------------------------------------------
# print("Generando análisis por dimensiones...")

# # Por cliente
# ventas_cliente = (
#     df.groupby(["id_cliente", "nombre_cliente"], as_index=False)
#     .agg(Ingreso_Total=("importe", "sum"), Ventas=("id_venta", "nunique"), Unidades=("cantidad", "sum"))
# )
# ventas_cliente["Ticket_Promedio"] = ventas_cliente["Ingreso_Total"] / ventas_cliente["Ventas"]

# # Por producto
# ventas_producto = (
#     df.groupby(["id_producto", "nombre_producto", "categoria"], as_index=False)
#     .agg(Ingreso_Total=("importe", "sum"), Unidades=("cantidad", "sum"))
# )
# ventas_producto["Precio_Medio"] = ventas_producto["Ingreso_Total"] / ventas_producto["Unidades"]

# # Por categoría
# ventas_categoria = (
#     df.groupby("categoria", as_index=False)
#     .agg(Ingreso_Total=("importe", "sum"), Unidades=("cantidad", "sum"))
# )
# ventas_categoria["Participación_%"] = ventas_categoria["Ingreso_Total"] / ventas_categoria["Ingreso_Total"].sum() * 100

# # Por ciudad
# ventas_ciudad = (
#     df.groupby("ciudad", as_index=False)
#     .agg(Ingreso_Total=("importe", "sum"), Unidades=("cantidad", "sum"), Ventas=("id_venta", "nunique"))
# )

# # -------------------------------------------------------------------
# # 5️⃣ EVOLUCIÓN TEMPORAL MENSUAL
# # -------------------------------------------------------------------
# df["Año_Mes"] = df["fecha"].dt.to_period("M")
# ventas_mensuales = df.groupby("Año_Mes", as_index=False)["importe"].sum().rename(columns={"importe": "Ingreso_Total"})
# ventas_mensuales["Año_Mes"] = ventas_mensuales["Año_Mes"].astype(str)

# # -------------------------------------------------------------------
# # 6️⃣ DISTRIBUCIÓN DE MEDIOS DE PAGO
# # -------------------------------------------------------------------
# medio_pago = (
#     df.groupby("medio_pago", as_index=False)
#     .agg(Ingreso_Total=("importe", "sum"), Ventas=("id_venta", "nunique"))
#     .sort_values("Ingreso_Total", ascending=False)
# )

# # -------------------------------------------------------------------
# # 7️⃣ GENERACIÓN DE GRÁFICOS CON MATPLOTLIB
# # -------------------------------------------------------------------
# print("Generando gráficos...")

# # Crear carpeta de salida si no existe
# os.makedirs("figuras", exist_ok=True)

# # Top 10 productos
# top10 = ventas_producto.sort_values("Ingreso_Total", ascending=False).head(10)
# plt.figure()
# plt.barh(top10["nombre_producto"], top10["Ingreso_Total"])
# plt.title("Top 10 productos por ingreso")
# plt.xlabel("Ingreso total")
# plt.tight_layout()
# plt.savefig("figuras/top10_productos.png", dpi=150)
# plt.close()

# # Categorías
# plt.figure()
# plt.bar(ventas_categoria["categoria"], ventas_categoria["Ingreso_Total"])
# plt.title("Ventas por categoría")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig("figuras/ventas_categoria.png", dpi=150)
# plt.close()

# # Ciudades
# plt.figure()
# plt.bar(ventas_ciudad["ciudad"], ventas_ciudad["Ingreso_Total"])
# plt.title("Ventas por ciudad")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig("figuras/ventas_ciudad.png", dpi=150)
# plt.close()

# # Evolución mensual
# plt.figure()
# plt.plot(ventas_mensuales["Año_Mes"], ventas_mensuales["Ingreso_Total"], marker="o")
# plt.title("Evolución mensual de ventas")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig("figuras/evolucion_mensual.png", dpi=150)
# plt.close()

# # Medios de pago
# plt.figure()
# plt.bar(medio_pago["medio_pago"], medio_pago["Ingreso_Total"])
# plt.title("Distribución por medio de pago")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig("figuras/medios_pago.png", dpi=150)
# plt.close()

# # Boxplot ticket por venta
# plt.figure()
# plt.boxplot(tickets.dropna(), labels=["Ticket por venta"])
# plt.title("Distribución de ticket por venta")
# plt.tight_layout()
# plt.savefig("figuras/boxplot_ticket.png", dpi=150)
# plt.close()

# print("Gráficos generados en la carpeta ./figuras")

# # -------------------------------------------------------------------
# # 8️⃣ EXPORTAR REPORTE A EXCEL
# # -------------------------------------------------------------------
# print("Exportando reporte a Excel...")

# with pd.ExcelWriter("reporte_analitico.xlsx", engine="xlsxwriter") as writer:
#     resumen_df.to_excel(writer, sheet_name="Resumen", index=False)
#     ventas_cliente.to_excel(writer, sheet_name="Por_Cliente", index=False)
#     ventas_producto.to_excel(writer, sheet_name="Por_Producto", index=False)
#     ventas_categoria.to_excel(writer, sheet_name="Por_Categoria", index=False)
#     ventas_ciudad.to_excel(writer, sheet_name="Por_Ciudad", index=False)
#     ventas_mensuales.to_excel(writer, sheet_name="Evolucion", index=False)
#     medio_pago.to_excel(writer, sheet_name="Medios_Pago", index=False)

# print("\n✅ Análisis completado.")
# print("Archivo generado: reporte_analitico.xlsx")
# print("Gráficos en carpeta: ./figuras/")
