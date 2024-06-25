import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json
import os
import warnings

# 2022
enero_2022_path = 'datasets_xlsx/2022/cuadro-insumos-medico-quirurgico-enero-2022.xlsx'
feb_2022_path = 'datasets_xlsx/2022/inventario-de-insumos-medico-quirurgico-febrero-2022.xlsx'
marzo_2022_path = 'datasets_xlsx/2022/inventario-de-insumo-medico-quirurgico-marzo-2022.xlsx'
abril_2022_path = 'datasets_xlsx/2022/inventario-de-medicamentos-abril-2022.xlsx'
mayo_2022_path = 'datasets_xlsx/2022/inventario-de-insumo-medico-quirurgico-de-mayo-2022.xlsx'
junio_2022_path = 'datasets_xlsx/2022/inventario-insumos-medico-quirurgico-junio-2022.xlsx'
julio_2022_path = 'datasets_xlsx/2022/inventario-de-medico-quirurgico-julio-2022.xlsx'
agost_2022_path = 'datasets_xlsx/2022/inventario-de-insumos-medico-quirurgico-agosto-2022.xlsx'
sept_2022_path = 'datasets_xlsx/2022/inventario-de-insumos-medico-quirurgico-septiembre-2022.xlsx'
oct_2022_path = 'datasets_xlsx/2022/inventario-insumos-medico-quirurgico-octubre-2022.xlsx'
nov_2022_path = 'datasets_xlsx/2022/inventario-medico-quirurgico-nov.-2022.xlsx'
dic_2022_path = 'datasets_xlsx/2022/inventario-medico-quirugico-dic.-2022.xlsx'
# 2023
enero_2023_path = 'datasets_xlsx/2023/cuadro-de-inventario-insumos-medico-quirurgico-enero-2023.xlsx'
feb_2023_path = 'datasets_xlsx/2023/cuadro-de-inventario-insumos-medico-quirurgico-febrero-2023.xlsx'
marzo_2023_path = 'datasets_xlsx/2023/cuadro-de-inventario-insumos-medico-quirurgico-marzo-2023.xlsx'
abril_2023_path = 'datasets_xlsx/2023/cuadro-de-inventario-insumos-medico-quirurgico-abril-2023.xlsx'
mayo_2023_path = 'datasets_xlsx/2023/cuadro-de-inventario-de-medico-quirurgico-de-mayo-2023.xlsx'
junio_2023_path = 'datasets_xlsx/2023/cuadro-de-inventario-de-medico-quirurgico-de-junio-2023.xlsx'
julio_2023_path = 'datasets_xlsx/2023/cuadro-de-inventario-de-medico-quirurgico-de-julio-2023.xlsx'
agost_2023_path = 'datasets_xlsx/2023/cuadro-de-inventario-medico-quirurgico-agosto-2023.xlsx'
sept_2023_path = 'datasets_xlsx/2023/inventario-mq-a-septiembre-2023.xlsx'
oct_2023_path = 'datasets_xlsx/2023/cuadro-de-inventario-medico-quirurgico-octubre-2023.xlsx'
nov_2023_path = 'datasets_xlsx/2023/cuadro-de-inventario-medico-quirurgico-nov.-2023.xlsx'
dic_2023_path = 'datasets_xlsx/2023/cuadro-de-inventario-medico-quirurgico-dic.-2023.xlsx'
# 2024
enero_2024_path = 'datasets_xlsx/2024/cuadro-de-inventario-insumos-medico-quirurgico-enero-2024.xlsx'
feb_2024_path = 'datasets_xlsx/2024/cuadro-de-inventario-de-insumos-medicos-quirurgico-feb.-2024.xlsx'
marzo_2024_path = 'datasets_xlsx/2024/inventario-insumos-medico-quirurgico-marzo-2024.xlsx'
abril_2024_path = 'datasets_xlsx/2024/cuadro-de-inventario-mq-abril-2024.xlsx'

# Definir las rutas de los archivos Excel (completa según tu ejemplo)
file_paths = [
    enero_2022_path, feb_2022_path, marzo_2022_path, abril_2022_path,
    mayo_2022_path, junio_2022_path, julio_2022_path, agost_2022_path,
    sept_2022_path, oct_2022_path, nov_2022_path, dic_2022_path,
    enero_2023_path, feb_2023_path, marzo_2023_path, abril_2023_path,
    mayo_2023_path, junio_2023_path, julio_2023_path, agost_2023_path,
    sept_2023_path, oct_2023_path, nov_2023_path, dic_2023_path,
    enero_2024_path, feb_2024_path, marzo_2024_path, abril_2024_path
]
months = [
    '2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01', '2022-06-01', 
    '2022-07-01', '2022-08-01', '2022-09-01', '2022-10-01', '2022-11-01', '2022-12-01',
    '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', 
    '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01', '2023-11-01', '2023-12-01',
    '2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01'
]

# Leer los archivos Excel en DataFrames y añadir una columna para el mes correspondiente
dfs = []
for file_path, month in zip(file_paths, months):
    df = pd.read_excel(file_path)
    df = df.iloc[:, [0, 1, -2]]  # Seleccionar las columnas por índice
    df['Fecha'] = month  # Añadir la columna de la fecha
    df = df.rename(columns={df.columns[2]: 'Inventario'})  # Renombrar la columna -2
    dfs.append(df)

# Concatenar todos los DataFrames
df_combined = pd.concat(dfs)

# Renombrar las columnas
df_combined.columns = ['Codigo', 'Insumo', 'Inventario', 'Fecha']

# Filtrar las filas con valores numéricos en 'Codigo'
df_insumos = df_combined[['Codigo', 'Insumo']]
df_insumos = df_insumos.dropna(subset=['Codigo'])
df_insumos['Codigo'] = df_insumos['Codigo'].apply(lambda x: ''.join(filter(str.isdigit, str(x))) if not str(x).isdigit() else x)

# Crear diccionario de código y nombre de producto
codigo_nombre_dict = df_insumos.set_index('Codigo')['Insumo'].to_dict()

# Exportar el diccionario a JSON
with open('json_files/codigo_nombre_productos.json', 'w') as json_file:
    json.dump(codigo_nombre_dict, json_file, indent=4)
    
# Reordenar las columnas
df_combined = df_combined[['Fecha', 'Codigo', 'Inventario']]

# Convertir la columna 'Fecha' a tipo datetime
df_combined['Fecha'] = pd.to_datetime(df_combined['Fecha'])

# Manejar NaN en la columna 'Inventario'
df_combined['Inventario'] = df_combined['Inventario'].fillna(0).astype(int)

# Contar el número de ceros por cada 'Codigo'
def count_zeros(df):
    return (df['Inventario'] == 0).sum()

# Filtrar los códigos con al menos 2 ceros
filtered_codigos = df_combined.groupby('Codigo').filter(lambda x: count_zeros(x) <= 2)

# Crear una tabla pivote
pivot_table = filtered_codigos.pivot_table(
    index='Codigo',
    columns='Fecha',
    values='Inventario',
    aggfunc='sum'
).fillna(0)

# Contar los ceros en cada fila
zero_counts = (pivot_table == 0).sum(axis=1)

# Filtrar las filas que tienen a lo sumo 2 ceros
filtered_pivot_table = pivot_table[zero_counts <= 2]

# Calcular el consumo (diferencia de inventarios entre meses consecutivos)
consumption = filtered_pivot_table.diff(axis=1).fillna(0)

# Entrenar y guardar los modelos SARIMAX
modelos = {}
predecibles = []
cantidad_adicional = 5000
avisos = []
output_dir = "modelos_sarimax"

for insumo in consumption.index:
    try:
        df_insumo = consumption.loc[insumo].dropna()

        if len(df_insumo) < 24:
            continue

        train_end = '2024-01-31'
        train_data = df_insumo[:train_end]
        test_data = df_insumo[train_end:]

        model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit(disp=False)

        codigo_producto = int(df_insumos[df_insumos['Insumo'] == insumo]['Codigo'].values[0])
        joblib.dump(results, f"{output_dir}/modelo_{codigo_producto}.pkl")

        predecibles.append({
            "Codigo": int(codigo_producto),  # Convertir a entero
            "Nombre": insumo,
            "ultimos_valores": [int(x) for x in df_insumo.tail(6)]  # Convertir a enteros
        })

        pred = results.get_forecast(steps=12)
        predicted_values = pred.predicted_mean

        for fecha, inventario in predicted_values.items():
            mes_anterior = (pd.to_datetime(fecha) - pd.DateOffset(months=1)).strftime('%Y-%m-01')
            mes_anterior = (pd.to_datetime(mes_anterior) + pd.DateOffset(days=-1)).strftime('%Y-%m-%d')
            cantidad_a_pedir = int(abs(inventario) + cantidad_adicional)  # Convertir a entero
            tipo = "REPOSICION" if cantidad_a_pedir > 0 else "CONSUMO"
            avisos.append({
                "codigo": int(codigo_producto),
                "insumo": insumo,
                "fecha_pedir_insumos": mes_anterior,
                "cantidad_a_pedir": cantidad_a_pedir,
                "mes_a_cubrir": (pd.to_datetime(fecha) + pd.DateOffset(days=-1)).strftime('%Y-%m-%d'),
                "Tipo": tipo
            })
    except Exception as e:
        print(f"Error al procesar el insumo {insumo}: {e}")

# Guardar el JSON de predecibles
with open('json_files/predecibles.json', 'w') as f:
    json.dump(predecibles, f, indent=4)

# Guardar el JSON de avisos
with open('json_files/requerimientos.json', 'w') as f:
    json.dump(avisos, f, indent=4)

print("Modelos y JSONs guardados exitosamente.")