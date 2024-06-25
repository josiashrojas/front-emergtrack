import os
import joblib
import subprocess
import json
import numpy as np
from flask import Flask, request, jsonify
from sklearn.metrics import mean_squared_error
from flask_cors import CORS

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

@app.route('/predict', methods=['POST'])
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

@app.route('/update_models', methods=['POST'])
def update_models():
    try:
        result = subprocess.run(['python', 'modelos.py'], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(result.stderr)
        return jsonify({'message': 'Modelos actualizados exitosamente'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_predecibles', methods=['GET'])
def get_predecibles():
    try:
        with open('modelos/predecibles.json', 'r') as f:
            predecibles = json.load(f)
        return jsonify(predecibles)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Cargar el JSON de requerimientos
with open('requerimientos.json', 'r') as file:
    requerimientos = json.load(file)

@app.route('/get_requerimientos', methods=['POST'])
def get_requerimientos():
    data = request.get_json()
    codigo = data.get('codigo')
    
    if not codigo:
        return jsonify({"error": "Código no proporcionado"}), 400

    # Filtrar los requerimientos por código
    filtered_requerimientos = [req for req in requerimientos if req['codigo'] == codigo]

    if not filtered_requerimientos:
        return jsonify({"error": "No se encontraron requerimientos para el código proporcionado"}), 404

    return jsonify(filtered_requerimientos)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)