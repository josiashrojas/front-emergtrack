import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import os

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

# Entrenar y guardar los modelos XGBoost
modelos = {}
scalers = {}

output_dir = "modelos"
os.makedirs(output_dir, exist_ok=True)

for codigo in consumption.index:
    df_codigo = consumption.loc[codigo].reset_index()
    df_codigo.columns = ['Fecha', 'Consumo']

    if df_codigo['Consumo'].isnull().sum() > 0:
        continue

    X = np.arange(len(df_codigo)).reshape(-1, 1)
    y = df_codigo['Consumo'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_scaled, y)

    modelos[codigo] = model
    scalers[codigo] = scaler

    joblib.dump(model, f"{output_dir}/xgb_model_{codigo}.pkl")
    joblib.dump(scaler, f"{output_dir}/scaler_{codigo}.pkl")

print("Modelos y scalers guardados para los códigos:")
print(list(modelos.keys()))
