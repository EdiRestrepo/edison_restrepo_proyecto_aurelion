# analisis_ventas.py
# -*- coding: utf-8 -*-
"""
Análisis estadístico y gráfico a partir de:
 - clientes.xlsx
 - productos.xlsx
 - ventas.xlsx
 - detalle_ventas.xlsx

Salida:
 - reporte_analitico.xlsx (múltiples hojas)
 - /figuras/*.png (gráficos en archivos)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Dict

# -----------------------------
# 0) Utilidades
# -----------------------------
def ensure_dir(path: str) -> None:
    """Crear carpeta si no existe."""
    if not os.path.exists(path):
        os.makedirs(path)

def read_excel_safe(fname: str) -> pd.DataFrame:
    """Leer un Excel y fallar con mensaje claro si no existe."""
    if not os.path.isfile(fname):
        raise FileNotFoundError(
            f"No se encontró '{fname}'. Asegúrate de que el script y los .xlsx están en la misma carpeta."
        )
    return pd.read_excel(fname)

# -----------------------------
# 1) Carga automática de archivos .xlsx
# -----------------------------
def load_data(
    clientes_path="clientes.xlsx",
    productos_path="productos.xlsx",
    ventas_path="ventas.xlsx",
    detalle_path="detalle_ventas.xlsx",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los cuatro archivos Excel desde el directorio actual.
    """
    clientes = read_excel_safe(clientes_path)
    productos = read_excel_safe(productos_path)
    ventas = read_excel_safe(ventas_path)
    detalle = read_excel_safe(detalle_path)
    return clientes, productos, ventas, detalle

# -----------------------------
# 2) Limpieza y unificación de datos
# -----------------------------
def clean_and_merge(
    clientes: pd.DataFrame,
    productos: pd.DataFrame,
    ventas: pd.DataFrame,
    detalle: pd.DataFrame,
) -> pd.DataFrame:
    """
    - Tipifica fechas y numéricos.
    - Calcula importe si falta.
    - Une detalle + ventas + productos + clientes en un único dataframe transaccional.
    - Resuelve colisiones de nombres (p.ej. nombre_producto duplicado).
    """
    # Tipificación de fechas
    if "fecha" in ventas.columns:
        ventas["fecha"] = pd.to_datetime(ventas["fecha"], errors="coerce")
    if "fecha_alta" in clientes.columns:
        clientes["fecha_alta"] = pd.to_datetime(clientes["fecha_alta"], errors="coerce")

    # Numéricos en detalle
    for col in ["cantidad", "precio_unitario", "importe"]:
        if col in detalle.columns:
            detalle[col] = pd.to_numeric(detalle[col], errors="coerce")

    # Si falta importe, calcularlo como cantidad * precio_unitario
    if "importe" not in detalle.columns or detalle["importe"].isna().any():
        if all(c in detalle.columns for c in ["cantidad", "precio_unitario"]):
            detalle["importe"] = detalle["cantidad"] * detalle["precio_unitario"]
        else:
            raise ValueError("Faltan columnas para calcular 'importe' (se requieren 'cantidad' y 'precio_unitario').")

    # Unión (join) de las tablas
    use_ventas = [c for c in ["id_venta", "fecha", "id_cliente", "medio_pago"] if c in ventas.columns]
    use_productos = [c for c in ["id_producto", "nombre_producto", "categoria"] if c in productos.columns]
    use_clientes = [c for c in ["id_cliente", "nombre_cliente", "ciudad"] if c in clientes.columns]

    df = detalle.merge(ventas[use_ventas], on="id_venta", how="left")
    df = df.merge(productos[use_productos], on="id_producto", how="left", suffixes=("_det", "_prod"))
    df = df.merge(clientes[use_clientes], on="id_cliente", how="left")

    # Resolver posibles duplicados de nombre_producto (si hubiera en 'detalle' y 'productos')
    # Tras el merge anterior, puede existir nombre_producto_det / nombre_producto_prod
    if "nombre_producto_det" in df.columns and "nombre_producto_prod" in df.columns:
        df["nombre_producto"] = df["nombre_producto_det"].fillna(df["nombre_producto_prod"])
    elif "nombre_producto" not in df.columns:
        # Si solo quedó una de las variantes:
        for candidate in ["nombre_producto_det", "nombre_producto_prod"]:
            if candidate in df.columns:
                df["nombre_producto"] = df[candidate]
                break

    # Asegurar columnas clave
    expected_cols = ["id_venta", "id_producto", "importe", "cantidad"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas necesarias en el dataset unificado: {missing}")

    return df

# -----------------------------
# 3) Cálculo de métricas
# -----------------------------
def compute_metrics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calcula:
    - Totales, promedios y desviaciones (resumen general).
    - Ventas por cliente.
    - Ventas por producto (con precio medio ponderado).
    - Ventas por categoría.
    - Ventas por ciudad.
    - Evolución mensual.
    - Distribución por medio de pago.
    Devuelve un dict con tablas (DataFrames).
    """
    # Resumen general (ticket por venta)
    ventas_por_transaccion = (
        df.groupby("id_venta", as_index=False)["importe"].sum().rename(columns={"importe": "importe_venta"})
    )
    tickets = ventas_por_transaccion["importe_venta"]

    resumen = {
        "periodo_min": [df["fecha"].min().date() if "fecha" in df.columns and not df["fecha"].isna().all() else None],
        "periodo_max": [df["fecha"].max().date() if "fecha" in df.columns and not df["fecha"].isna().all() else None],
        "n_ventas": [ventas_por_transaccion["id_venta"].nunique()],
        "n_lineas_detalle": [len(df)],
        "unidades_vendidas": [int(df["cantidad"].sum())],
        "ingreso_total": [float(df["importe"].sum())],
        "ticket_promedio_por_venta": [float(tickets.mean()) if len(tickets) > 0 else np.nan],
        "desv_std_ticket": [float(tickets.std(ddof=1)) if len(tickets) > 1 else np.nan],
        "importe_promedio_por_linea": [float(df["importe"].mean()) if len(df) > 0 else np.nan],
    }
    resumen_df = pd.DataFrame(resumen)

    # Por cliente
    keys_cliente = [c for c in ["id_cliente", "nombre_cliente"] if c in df.columns]
    por_cliente = (
        df.groupby(keys_cliente, as_index=False)
        .agg(ingreso_total=("importe", "sum"), n_ventas=("id_venta", "nunique"), unidades=("cantidad", "sum"))
        .sort_values("ingreso_total", ascending=False)
    )
    por_cliente["ticket_promedio_cliente"] = por_cliente["ingreso_total"] / por_cliente["n_ventas"].replace(0, np.nan)

    # Por producto
    keys_producto = [c for c in ["id_producto", "nombre_producto", "categoria"] if c in df.columns]
    por_producto = (
        df.groupby(keys_producto, as_index=False)
        .agg(ingreso_total=("importe", "sum"), unidades=("cantidad", "sum"))
        .sort_values("ingreso_total", ascending=False)
    )
    por_producto["precio_medio_vendido"] = por_producto["ingreso_total"] / por_producto["unidades"].replace(0, np.nan)

    # Por categoría
    if "categoria" in df.columns:
        por_categoria = (
            df.groupby(["categoria"], as_index=False)
            .agg(ingreso_total=("importe", "sum"), unidades=("cantidad", "sum"))
            .sort_values("ingreso_total", ascending=False)
        )
        por_categoria["participacion_%"] = por_categoria["ingreso_total"] / por_categoria["ingreso_total"].sum() * 100
    else:
        por_categoria = pd.DataFrame(columns=["categoria", "ingreso_total", "unidades", "participacion_%"])

    # Por ciudad
    if "ciudad" in df.columns:
        por_ciudad = (
            df.groupby(["ciudad"], as_index=False)
            .agg(ingreso_total=("importe", "sum"), unidades=("cantidad", "sum"), n_ventas=("id_venta", "nunique"))
            .sort_values("ingreso_total", ascending=False)
        )
    else:
        por_ciudad = pd.DataFrame(columns=["ciudad", "ingreso_total", "unidades", "n_ventas"])

    # Evolución mensual
    if "fecha" in df.columns:
        df_tmp = df.copy()
        df_tmp["anio_mes"] = df_tmp["fecha"].dt.to_period("M")
        ventas_mensuales = (
            df_tmp.groupby("anio_mes", as_index=False)["importe"].sum().rename(columns={"importe": "ingreso_total"})
        )
        ventas_mensuales["anio_mes"] = ventas_mensuales["anio_mes"].astype(str)
    else:
        ventas_mensuales = pd.DataFrame(columns=["anio_mes", "ingreso_total"])

    # Medios de pago
    if "medio_pago" in df.columns:
        medio_pago_dist = (
            df.groupby("medio_pago", as_index=False)
            .agg(ingreso_total=("importe", "sum"), n_ventas=("id_venta", "nunique"))
            .sort_values("ingreso_total", ascending=False)
        )
    else:
        medio_pago_dist = pd.DataFrame(columns=["medio_pago", "ingreso_total", "n_ventas"])

    return dict(
        resumen=resumen_df,
        por_cliente=por_cliente,
        por_producto=por_producto,
        por_categoria=por_categoria,
        por_ciudad=por_ciudad,
        ventas_mensuales=ventas_mensuales,
        medio_pago_dist=medio_pago_dist,
        tickets=tickets.to_frame(),  # guardo tickets por si quiero graficar boxplot
    )

# -----------------------------
# 4) Generación de gráficos con matplotlib
# -----------------------------
def make_plots(tables: Dict[str, pd.DataFrame], out_dir="figuras") -> None:
    """
    Genera y guarda gráficos en la carpeta /figuras.
    Reglas: una figura por gráfico, sin estilos/colores específicos.
    """
    ensure_dir(out_dir)

    # a) Top 10 productos por ingreso (barra horizontal)
    por_producto = tables["por_producto"].head(10)
    plt.figure()
    plt.barh(por_producto["nombre_producto"][::-1], por_producto["ingreso_total"][::-1])
    plt.xlabel("Ingreso total")
    plt.ylabel("Producto")
    plt.title("Top 10 productos por ingreso")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top10_productos.png"), dpi=150)
    plt.close()

    # b) Ventas por categoría (barra)
    por_categoria = tables["por_categoria"]
    if not por_categoria.empty:
        plt.figure()
        plt.bar(por_categoria["categoria"], por_categoria["ingreso_total"])
        plt.xlabel("Categoría")
        plt.ylabel("Ingreso total")
        plt.title("Ventas por categoría")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ventas_por_categoria.png"), dpi=150)
        plt.close()

    # c) Ventas por ciudad (barra)
    por_ciudad = tables["por_ciudad"]
    if not por_ciudad.empty:
        plt.figure()
        plt.bar(por_ciudad["ciudad"], por_ciudad["ingreso_total"])
        plt.xlabel("Ciudad")
        plt.ylabel("Ingreso total")
        plt.title("Ventas por ciudad")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ventas_por_ciudad.png"), dpi=150)
        plt.close()

    # d) Evolución mensual (línea)
    ventas_mensuales = tables["ventas_mensuales"]
    if not ventas_mensuales.empty:
        plt.figure()
        plt.plot(ventas_mensuales["anio_mes"], ventas_mensuales["ingreso_total"], marker="o")
        plt.xlabel("Año-Mes")
        plt.ylabel("Ingreso total")
        plt.title("Evolución mensual de ventas")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "evolucion_mensual.png"), dpi=150)
        plt.close()

    # e) Medios de pago (barra)
    medio_pago = tables["medio_pago_dist"]
    if not medio_pago.empty:
        plt.figure()
        plt.bar(medio_pago["medio_pago"], medio_pago["ingreso_total"])
        plt.xlabel("Medio de pago")
        plt.ylabel("Ingreso total")
        plt.title("Distribución de ingresos por medio de pago")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "medios_de_pago.png"), dpi=150)
        plt.close()

    # f) Boxplot de tickets por venta
    tickets = tables["tickets"]
    if not tickets.empty:
        plt.figure()
        plt.boxplot(tickets["importe_venta"].dropna(), vert=True, labels=["Ticket por venta"])
        plt.ylabel("Importe")
        plt.title("Distribución de ticket por venta")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "boxplot_ticket.png"), dpi=150)
        plt.close()

# -----------------------------
# 5) Exportación a Excel con múltiples hojas
# -----------------------------
def export_to_excel(tables: Dict[str, pd.DataFrame], out_path="reporte_analitico.xlsx") -> None:
    """
    Exporta todas las tablas a un Excel con hojas separadas.
    """
    engine = "xlsxwriter"  # si no lo tienes, usa openpyxl
    with pd.ExcelWriter(out_path, engine=engine) as writer:
        tables["resumen"].to_excel(writer, sheet_name="resumen", index=False)
        tables["por_cliente"].to_excel(writer, sheet_name="por_cliente", index=False)
        tables["por_producto"].to_excel(writer, sheet_name="por_producto", index=False)
        tables["por_categoria"].to_excel(writer, sheet_name="por_categoria", index=False)
        tables["por_ciudad"].to_excel(writer, sheet_name="por_ciudad", index=False)
        tables["ventas_mensuales"].to_excel(writer, sheet_name="ventas_mensuales", index=False)
        tables["medio_pago_dist"].to_excel(writer, sheet_name="medio_pago", index=False)

# -----------------------------
# 6) Punto de entrada (main)
# -----------------------------
def main():
    """
    Pipeline completo:
    1) Carga de .xlsx
    2) Limpieza + unificación
    3) Cálculo de métricas (totales, promedios, desviaciones)
       - Por cliente, producto, categoría, ciudad
       - Evolución mensual
       - Medios de pago
    4) Gráficos (matplotlib) -> carpeta /figuras
    5) Exportación a Excel -> reporte_analitico.xlsx
    """
    print(">> Cargando datos .xlsx ...")
    clientes, productos, ventas, detalle = load_data()

    print(">> Limpiando y unificando datos ...")
    df = clean_and_merge(clientes, productos, ventas, detalle)

    print(">> Calculando métricas ...")
    tables = compute_metrics(df)

    print(">> Generando gráficos (carpeta ./figuras) ...")
    make_plots(tables, out_dir="figuras")

    print(">> Exportando reporte a Excel ...")
    export_to_excel(tables, out_path="reporte_analitico.xlsx")

    print("\n¡Listo!")
    print(" - Reporte Excel:  ./reporte_analitico.xlsx")
    print(" - Figuras (PNG):  ./figuras/*.png")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]", str(e))
        sys.exit(1)
