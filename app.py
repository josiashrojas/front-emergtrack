import os
import joblib
import subprocess
import json
import numpy as np
from flask import Flask, request, jsonify
from sklearn.metrics import mean_squared_error
from flask_cors import CORS
import pandas as pd  # Importar pandas para manejar las fechas
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Cargar los modelos y scalers
modelos = {}
scalers = {}

model_files = [f for f in os.listdir('modelos') if 'xgb_model' in f]
scaler_files = [f for f in os.listdir('modelos') if 'scaler' in f]

for model_file in model_files:
    codigo = model_file.replace('xgb_model_', '').replace('.pkl', '')
    modelos[codigo] = joblib.load(f"modelos/{model_file}")

for scaler_file in scaler_files:
    codigo = scaler_file.replace('scaler_', '').replace('.pkl', '')
    scalers[codigo] = joblib.load(f"modelos/{scaler_file}")

# Cargar los modelos SARIMAX desde archivos
output_dir = "modelos_sarimax"
model_files = [f for f in os.listdir(output_dir) if f.startswith('modelo_') and f.endswith('.pkl')]

for model_file in model_files:
    codigo_producto = int(model_file.split('_')[1].split('.')[0])
    modelos[codigo_producto] = joblib.load(os.path.join(output_dir, model_file))

# Cargar el archivo JSON con los nombres de los insumos
with open('json_files/codigo_nombre_productos.json', 'r') as json_file:
    codigo_nombre_dict = json.load(json_file)

# --------------------------- Funciones --------------------------- #
def is_valid_date(date_str):
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False
# --------------------------- --------- --------------------------- #

@app.route('/predict', methods=['GET'])
def predict():
    data = request.get_json()
    codigo = data['codigo']
    periodos = data['periodos']
    last_known_values = data.get('last_known_values', [])  # Obtener last_known_values si está presente, de lo contrario, usar lista vacía

    if codigo not in modelos:
        return jsonify({'error': 'Código no encontrado - No es posible predecir producto'}), 404

    model = modelos[codigo]
    scaler = scalers[codigo]

    # Crear características para predicción futura
    last_index = len(last_known_values)
    future_X = np.arange(last_index + 1, last_index + 1 + periodos).reshape(-1, 1)
    future_X_scaled = scaler.transform(future_X)

    # Hacer predicciones
    predictions = model.predict(future_X_scaled)

    # Calcular MSE si hay valores reales proporcionados
    mse = None
    if len(last_known_values) >= periodos:
        actual_values = np.array(last_known_values[-periodos:])
        mse = mean_squared_error(actual_values, predictions)

    return jsonify({
        'codigo': codigo,
        'predicciones': predictions.tolist(),
        'mse': mse
    })

@app.route('/update_models', methods=['GET'])
def update_models():
    try:
        result = subprocess.run(['python', 'train_model.py'], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(result.stderr)
        return jsonify({'message': 'Modelos actualizados exitosamente'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_predecibles', methods=['GET'])
def get_predecibles():
    try:
        with open('json_files/predecibles.json', 'r') as f:
            predecibles = json.load(f)
        return jsonify(predecibles)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_requerimientos', methods=['GET'])
def get_requerimientos():
    data = request.get_json()
    codigo = data.get('codigo')
    fecha_base = data.get('fecha_base')

    if not codigo:
        return jsonify({"error": "Código no proporcionado"}), 400

    if fecha_base and not is_valid_date(fecha_base):
        return jsonify({"error": "Fecha base no válida"}), 400

    codigo = int(codigo)

    if codigo not in modelos:
        return jsonify({"error": "No se encontró el modelo para el producto proporcionado"}), 404

    modelo = modelos[codigo]

    # Obtener la última fecha de entrenamiento del modelo
    end_date = modelo.data.dates[-1]
    periods = 12  # Número de periodos a predecir

    # Realizar predicción
    pred = modelo.get_forecast(steps=periods)
    predicted_values = pred.predicted_mean

    # Definir la cantidad adicional de seguridad
    cantidad_adicional = 100

    # Lista para almacenar los avisos
    avisos = []

    # Generar avisos para cada mes con inventario negativo
    for fecha, inventario in predicted_values.items():
        mes_anterior = (pd.to_datetime(fecha) - pd.DateOffset(months=1)).strftime('%Y-%m-01')
        mes_anterior = (pd.to_datetime(mes_anterior) + pd.DateOffset(days=-1)).strftime('%Y-%m-%d')
        nombre_insumo = codigo_nombre_dict.get(str(codigo), f"Insumo_{codigo}")
        cantidad_a_pedir = int(abs(inventario) + cantidad_adicional)  # Convertir a entero
        tipo = "APROVISIONAMIENTO" if inventario > 0 else "CONSUMO"
        aviso = {
            "codigo": codigo,
            "insumo": nombre_insumo,  # Aquí puedes agregar el nombre real del insumo si lo tienes disponible
            "fecha_pedir_insumos": mes_anterior,
            "cantidad_a_pedir": cantidad_a_pedir,
            "mes_a_cubrir": (pd.to_datetime(fecha) + pd.DateOffset(days=-1)).strftime('%Y-%m-%d'),
            "tipo": tipo
        }
        avisos.append(aviso)

    # Filtrar los requerimientos por fecha_base si se proporciona
    if fecha_base:
        fecha_base = pd.to_datetime(fecha_base)
        avisos = [aviso for aviso in avisos if pd.to_datetime(aviso['mes_a_cubrir']) >= fecha_base]

    if not avisos:
        return jsonify({"error": "No se encontraron requerimientos desde la fecha proporcionada"}), 404

    return jsonify(avisos)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)