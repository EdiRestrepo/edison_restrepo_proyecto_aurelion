#pip install pandas
#pip install openpyxl
import pandas as pd

# Cargar archivos
clientes = pd.read_excel("clientes.xlsx")
ventas = pd.read_excel("ventas.xlsx")
detalle = pd.read_excel("detalle_ventas.xlsx")
productos = pd.read_excel("productos.xlsx")

# Unir detalle con ventas
df = detalle.merge(ventas[['id_venta']], on='id_venta', how='left')

#Estadísticas Básicas
""" def estadisticas_basicas(df):
    numericas = ['cantidad', 'precio_unitario', 'importe']
    resumen = {}
    for col in numericas:
        resumen[col] = {
            'media': df[col].mean(),
            # 'mediana': df[col].median(),
            'moda': df[col].mode()[0],
            'std': df[col].std(),
            'rango': df[col].max() - df[col].min()
        }
    return pd.DataFrame(resumen).round(2)
print("Estadísticas Básicas:", estadisticas_basicas(df), sep="\n") """

def estadisticas_basicas(df):
    numericas = ['cantidad', 'precio_unitario', 'importe']
    resumen = {}
    for col in numericas:
        modo = df[col].mode()
        modo_val = modo.iloc[0] if not modo.empty else pd.NA
        resumen[col] = {
            'media': df[col].mean(),
            'mediana': df[col].median(),
            'moda': modo_val,
            'std': df[col].std(),
            'rango': df[col].max() - df[col].min()
        }
    stats = pd.DataFrame(resumen).round(2)  # price_unitario e importe quedan con 2 decimales
    # convertir 'cantidad' a entero (nullable) redondeando primero
    if 'cantidad' in stats.columns:
        stats['cantidad'] = stats['cantidad'].round(0).astype('Int64')
    return stats
print("Estadísticas Básicas:", estadisticas_basicas(df), sep="\n")

#/ ...existing code...
def producto_mas_vendido(df):
    # Agrupa por nombre de producto si existe, si no usa id_producto
    key = 'nombre_producto' if 'nombre_producto' in df.columns else 'id_producto'
    vendidos = df.groupby(key)['cantidad'].sum()
    if vendidos.empty:
        return pd.NA, 0
    producto = vendidos.idxmax()
    cantidad = int(vendidos.max())
    return producto, cantidad

# Ejemplo de uso
producto, cantidad = producto_mas_vendido(df)
print(f"Producto más vendido: {producto} (cantidad total: {cantidad})")
#/ ...existing code...

# IDENTIFICACIÓN DEL TIPO DE DISTRIBUCIÓN DE VARIABLES
# ...existing code...
# Nueva celda: análisis automático del tipo de distribución
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns

def detect_modality(series, bw_method='scott', peaks_prominence=0.05):
    """Devuelve número de modos estimados usando KDE y detección de picos."""
    x = series.dropna().values
    if len(x) < 5:
        return 0
    kde = stats.gaussian_kde(x, bw_method=bw_method)
    grid = np.linspace(x.min(), x.max(), 1000)
    kde_vals = kde(grid)
    # normalizar para usar prominencia relativa
    prom_thresh = peaks_prominence * (kde_vals.max() - kde_vals.min())
    peaks, _ = find_peaks(kde_vals, prominence=prom_thresh)
    return len(peaks)

def is_uniform(series, alpha=0.05):
    """Test KS contra distribución uniforme en el rango [min, max]."""
    x = series.dropna().values
    if len(x) < 10:
        return False
    # Escalar a [0,1]
    a, b = x.min(), x.max()
    if a == b:
        return False
    x_u = (x - a) / (b - a)
    stat, p = stats.kstest(x_u, 'uniform')
    return p > alpha

def analyze_distribution(df, col, plot=True):
    s = df[col].dropna()
    # Categórica/nominal
    if not pd.api.types.is_numeric_dtype(s) or s.nunique() <= 20:
        return {'column': col, 'type': 'nominal/categórica', 'detail': f'unique={s.nunique()}'}
    # Estadísticos básicos
    skewness = stats.skew(s)
    kurt = stats.kurtosis(s)
    normal_p = None
    try:
        normal_stat, normal_p = stats.normaltest(s)  # D'Agostino
    except Exception:
        normal_p = None
    # Modalidad
    modos = detect_modality(s)
    # Uniformidad
    uniform = is_uniform(s)
    # Clasificación simple
    if uniform:
        dist_type = 'uniforme'
    elif modos >= 2:
        dist_type = 'bimodal' if modos == 2 else 'multimodal'
    elif normal_p is not None and normal_p > 0.05 and abs(skewness) < 0.5:
        dist_type = 'normal (aprox.)'
    else:
        if skewness > 0.5:
            dist_type = 'sesgada a la derecha (positiva)'
        elif skewness < -0.5:
            dist_type = 'sesgada a la izquierda (negativa)'
        else:
            dist_type = 'asimétrica leve / unimodal'
    result = {
        'column': col,
        'type': dist_type,
        'skewness': float(skewness),
        'kurtosis': float(kurt),
        'normaltest_p': float(normal_p) if normal_p is not None else None,
        'modality_peaks': int(modos),
        'is_uniform': bool(uniform)
    }
    # Plot opcional
    if plot:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        sns.histplot(s, kde=False, bins=30)
        plt.title(f'Histograma: {col}')
        plt.subplot(1,2,2)
        sns.kdeplot(s, fill=True)
        plt.title(f'KDE: {col} — {dist_type}')
        plt.tight_layout()
        plt.show()
    return result

# Uso: analizar las columnas numéricas principales
for col in ['cantidad', 'precio_unitario', 'importe']:
    print(analyze_distribution(df, col, plot=True))
# ...existing code..
# 
## ANÁLISIS DE CORRELACIÓN ENTRE VARIABLES.
import itertools
from scipy import stats

def _pairwise_pvalues(df, cols, method='pearson'):
    """Devuelve DataFrame de p-values para correlaciones entre cols."""
    pvals = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    for i, j in itertools.combinations(cols, 2):
        x = df[i].dropna()
        y = df[j].dropna()
        # alinear por índices comunes
        common_idx = x.index.intersection(y.index)
        if len(common_idx) < 3:
            p = np.nan
        else:
            x0, y0 = x.loc[common_idx], y.loc[common_idx]
            try:
                if method == 'pearson':
                    _, p = stats.pearsonr(x0, y0)
                elif method == 'spearman':
                    _, p = stats.spearmanr(x0, y0)
                elif method == 'kendall':
                    _, p = stats.kendalltau(x0, y0)
                else:
                    _, p = stats.pearsonr(x0, y0)
            except Exception:
                p = np.nan
        pvals.at[i, j] = p
        pvals.at[j, i] = p
    np.fill_diagonal(pvals.values, 0.0)
    return pvals

def analyze_correlations(df, method='pearson', top_n=10, threshold=0.5, plot=True):
    """
    Calcula matriz de correlación y p-values entre variables numéricas.
    Imprime resumen y dibuja heatmap + scatter para los pares más correlacionados.
    """
    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] < 2:
        print("No hay suficientes columnas numéricas para analizar correlaciones.")
        return {}
    cols = num.columns.tolist()
    corr = num.corr(method=method)
    pvals = _pairwise_pvalues(num, cols, method=method)

    print(f"Corr ({method}):")
    print(corr.round(3))
    print("\nP-values (pearson/specified):")
    print(pvals.round(4))

    # Extraer pares ordenados por |corr|
    pairs = []
    for i, j in itertools.combinations(cols, 2):
        val = corr.at[i, j]
        pv = pvals.at[i, j]
        pairs.append((i, j, val, pv))
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    print(f"\nTop {top_n} pares por |correlación| (umbral {threshold}):")
    cnt = 0
    for a, b, v, p in pairs_sorted[:top_n]:
        sig = "p<{:.3f}".format(p) if (not np.isnan(p) and p < 0.05) else f"p={p if not np.isnan(p) else 'NA'}"
        mark = "(>=th)" if abs(v) >= threshold else ""
        print(f"{a} ⟷ {b}: corr={v:.3f} {mark} {sig}")
        cnt += 1

    if plot:
        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f"Heatmap de correlación ({method})")
        plt.tight_layout()
        plt.show()

        # Scatter para los top pares significativos o con mayor |corr|
        to_plot = [ (a,b) for a,b,_,_ in pairs_sorted if abs(_)>=threshold ]
        if not to_plot:
            to_plot = [(a,b) for a,b,_,_ in pairs_sorted[:min(4, len(pairs_sorted))]]
        n = len(to_plot)
        if n>0:
            plt.figure(figsize=(5*n,4))
            for idx, (a,b) in enumerate(to_plot,1):
                plt.subplot(1,n,idx)
                sns.scatterplot(x=num[a], y=num[b], alpha=0.6)
                sns.regplot(x=num[a], y=num[b], scatter=False, color='r', ci=None)
                plt.xlabel(a); plt.ylabel(b)
                plt.title(f"{a} vs {b}\n corr={corr.at[a,b]:.2f}")
            plt.tight_layout()
            plt.show()

    return {'corr': corr, 'pvalues': pvals, 'pairs_sorted': pairs_sorted}

# Ejemplos de uso:
print("\n--- Análisis de correlaciones en detalle (detalle_ventas) ---")
res_corr = analyze_correlations(detalle, method='pearson', top_n=10, threshold=0.5, plot=True)

# Correlaciones sobre ventas agregadas por id_venta
ventas_por_venta = detalle.groupby('id_venta', as_index=False).agg(
    venta_importe=('importe','sum'),
    venta_cantidad=('cantidad','sum')
)
print("\n--- Correlaciones en ventas agregadas (id_venta) ---")
_ = analyze_correlations(ventas_por_venta, method='spearman', top_n=6, threshold=0.3, plot=True)

## DETECCIÓN DE OUTLIERS (Valores Atípicos)

import numpy as np
from scipy import stats

def outliers_iqr(series, factor=1.5):
    s = series.dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    mask = (series < lower) | (series > upper)
    return mask.fillna(False), lower, upper

def outliers_zscore(series, threshold=3.0):
    s = series.dropna()
    if s.shape[0] < 2:
        return pd.Series(False, index=series.index), None, None
    z = np.abs(stats.zscore(s))
    full_z = pd.Series(index=s.index, data=z)
    mask = series.index.to_series().apply(lambda i: full_z.get(i, np.nan)).fillna(0) > threshold
    return mask, -threshold, threshold

def outliers_mad(series, threshold=3.5):
    s = series.dropna()
    med = s.median()
    mad = (np.abs(s - med)).median()
    if mad == 0:
        return pd.Series(False, index=series.index), med, mad
    modified_z = 0.6745 * (s - med) / mad
    mask = np.abs(modified_z) > threshold
    full_mask = pd.Series(False, index=series.index)
    full_mask.loc[s.index] = mask
    return full_mask, med, mad

def summarize_outliers(df, cols=None, method='iqr', **kwargs):
    """Devuelve resumen de outliers por columna y muestra ejemplos."""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = {}
    for col in cols:
        if col not in df.columns:
            continue
        ser = df[col]
        if method == 'zscore':
            mask, lo, hi = outliers_zscore(ser, **kwargs)
        elif method == 'mad':
            mask, lo, hi = outliers_mad(ser, **kwargs)
        else:
            mask, lo, hi = outliers_iqr(ser, **kwargs)
        n_out = int(mask.sum())
        pct = n_out / float(len(ser.dropna())) if len(ser.dropna())>0 else 0
        examples = df.loc[mask, col].head(5).tolist()
        summary[col] = {'n_outliers': n_out, 'pct': round(pct,4), 'examples': examples, 'bounds': (lo, hi)}
        print(f"{col}: {n_out} outliers ({pct:.2%}). Ejemplos: {examples}  bounds={ (lo,hi) }")
    return summary

def remove_outliers(df, cols=None, method='iqr', how='drop', replace_with=None, **kwargs):
    """
    how: 'drop' (elimina filas con outlier en cualquier col),
         'cap' (recorta valores al límite),
         'replace' (reemplaza outlier por replace_with, default NaN)
    """
    df2 = df.copy()
    if cols is None:
        cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    masks = []
    for col in cols:
        if method == 'zscore':
            mask, lo, hi = outliers_zscore(df2[col], **kwargs)
        elif method == 'mad':
            mask, lo, hi = outliers_mad(df2[col], **kwargs)
        else:
            mask, lo, hi = outliers_iqr(df2[col], **kwargs)
        masks.append(mask)
        if how == 'cap' and lo is not None and hi is not None:
            df2.loc[df2[col] < lo, col] = lo
            df2.loc[df2[col] > hi, col] = hi
        elif how == 'replace':
            rep = replace_with if replace_with is not None else np.nan
            df2.loc[mask, col] = rep
    if how == 'drop':
        combined = pd.concat(masks, axis=1).any(axis=1)
        df2 = df2.loc[~combined].reset_index(drop=True)
    return df2

# Ejemplos rápidos de uso:
print("\n--- Resumen outliers (IQR) en detalle ---")
summarize_outliers(detalle, cols=['cantidad','precio_unitario','importe'], method='iqr', factor=1.5)

# Detectar y mostrar outliers en ventas agregadas por id_venta
ventas_por_venta = detalle.groupby('id_venta', as_index=False).agg(
    venta_importe=('importe','sum'),
    venta_cantidad=('cantidad','sum')
)
print("\n--- Resumen outliers en ventas agregadas (IQR) ---")
summarize_outliers(ventas_por_venta, cols=['venta_importe','venta_cantidad'], method='iqr', factor=1.5)

# Ejemplo: eliminar filas con outliers por IQR
ventas_sin_outliers = remove_outliers(ventas_por_venta, cols=['venta_importe','venta_cantidad'], method='iqr', how='drop', factor=1.5)
print(f"\nOriginal ventas_por_venta: {len(ventas_por_venta)} filas, sin outliers: {len(ventas_sin_outliers)} filas")

# Opcional: graficar antes/después
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.boxplot(x=ventas_por_venta['venta_importe'])
    plt.title('Antes: venta_importe')
    plt.subplot(1,2,2)
    sns.boxplot(x=ventas_sin_outliers['venta_importe'])
    plt.title('Después: venta_importe (sin outliers)')
    plt.tight_layout()
    plt.show()
except Exception:
    pass



# # Cargar archivos
# import pandas as pd

# clientes = pd.read_excel("clientes.xlsx")
# ventas = pd.read_excel("ventas.xlsx")
# detalle = pd.read_excel("detalle_ventas.xlsx")
# productos = pd.read_excel("productos.xlsx")

# # Unir detalle con ventas
# df = detalle.merge(ventas[['id_venta', 'fecha']], on='id_venta', how='left')

# #Estadísticas Básicas
# def estadisticas_basicas(df):
#     numericas = ['cantidad', 'precio_unitario', 'importe']
#     resumen = {}
#     for col in numericas:
#         resumen[col] = {
#             'media': df[col].mean(),
#             'mediana': df[col].median(),
#             'moda': df[col].mode()[0],
#             'std': df[col].std(),
#             'rango': df[col].max() - df[col].min()
#         }
#     return pd.DataFrame(resumen)

# print("Estadísticas Básicas:\n", estadisticas_basicas(df))
# # print("df:\n", df)

# # import pandas as pd
# # from pathlib import Path

# # DATA_DIR = Path(__file__).resolve().parent
# # df = pd.read_excel(DATA_DIR / "clientes.xlsx", engine="openpyxl")
# # print(df.shape, df.columns.tolist())

# # # Paso 1: Cargar datos
# # clientes = pd.read_excel("clientes.xlsx")
# # productos = pd.read_excel("productos.xlsx")
# # detalle = pd.read_excel("detalle_ventas.xlsx")
# # ventas = pd.read_excel("ventas.xlsx")

# # # Paso 2: Unir datasets
# # ventas_clientes = ventas.merge(clientes, on="id_cliente")
# # detalle_productos = detalle.merge(productos, on="id_producto")
# # ventas_detalle = detalle_productos.merge(ventas_clientes, on="id_venta")

# # # Paso 3: Métricas
# # ventas_por_producto = ventas_detalle.groupby("nombre_producto_x")["importe"].sum()
# # ventas_por_cliente = ventas_detalle.groupby("nombre_cliente_x")["importe"].sum()
# # ventas_por_medio = ventas_detalle["medio_pago"].value_counts()
# # ventas_por_categoria = ventas_detalle.groupby("categoria")["importe"].sum()
# # ventas_por_ciudad = ventas_detalle.groupby("ciudad")["importe"].sum()

# # # usando ventas_detalle existente
# # ventas_detalle["importe"] = pd.to_numeric(ventas_detalle["importe"], errors="coerce").fillna(0)
# # top3 = ventas_detalle.groupby("nombre_cliente_x")["importe"].sum().nlargest(3)
# # print("Top 3 Clientes por Ventas:\n", top3)

# # print("Ventas por Producto:\n", ventas_por_producto)
# # print("\nVentas por Cliente:\n", ventas_por_cliente)
# # print("\nVentas por Medio de Pago:\n", ventas_por_medio)
# # print("\nVentas por Categoría:\n", ventas_por_categoria)
# # print("\nVentas por Ciudad:\n", ventas_por_ciudad)
