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
- id_cliente: Identificador único.
- nombre_cliente:  Nombre completo del cliente.
- email: Correo electrónico.
- ciudad: Ciudad de residencia.
- fecha_alta: Fecha de alta del cliente.

# Productos
- id_producto: Identificador único.
- nombre_producto: Nombre del producto.
- categoría: clasificación del producto.
- precio_unitario: precio unitario de producto.

# Ventas
- id_venta: Identificador único.
- fecha: Fecha de compra.
- id_cliente: Relación con Cliente
- medio_pago: Forma de pago

# TIPOS DE DATOS
# Clientes
- id_cliente: INT
- nombre_cliente: VARCHAR(100).
- email: VARCHAR(100), UNIQUE.
- ciudad: VARCHAR(50).
- fecha_alta:  DATE.

# Productos
- id_producto: INT 
- nombre_producto: VARCHAR(100).
- categoria: VARCHAR(50).
- precio_unitario: DECIMAL(10,2).

# Ventas
- id_venta: INT 
- fecha: DATE
- id_cliente: INT
- medio_pago: VARCHAR(20).

# ESCALA
**Escala actual:**
- Clientes: decenas a cientos.
- Productos: catálogo pequeño.
- Ventas: cientos a miles de registros.

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





