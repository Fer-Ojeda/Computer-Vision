# api/api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from code_processor import procesar_anaquel_para_api # Tu script adaptado
import datetime

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

# Esta ruta es para el chequeo inicial en api.py. La ruta real del modelo
# se define en code_processor.py o se pasa a procesar_anaquel_para_api.
# Para consistencia con tu code.py, podrías usar:
# PATH_MODELO_YOLO_PRINCIPAL = r"C:\Users\ferna\Documents\Hack\full-code\best(2).pt"
# O, si el modelo está dentro de la carpeta api/ (recomendado para la API):
PATH_MODELO_YOLO_PRINCIPAL = os.path.join(BASE_DIR, "runs", "best(2).pt") # Ajusta si tu "best(2).pt" está aquí

if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)
if not os.path.exists(PATH_MODELO_YOLO_PRINCIPAL):
    print(f"ADVERTENCIA API: El archivo del modelo YOLO principal no se encontró en {PATH_MODELO_YOLO_PRINCIPAL}")
    print("Verifica la ruta en api/api.py.")

@app.route('/api/analizar-anaquel', methods=['POST'])
def analizar_anaquel_endpoint():
    if 'imagen' not in request.files:
        return jsonify({"error": "No se encontró el archivo de imagen"}), 400
    if 'anaquel_id' not in request.form:
        return jsonify({"error": "No se especificó el ID del anaquel"}), 400

    imagen_file = request.files['imagen']
    anaquel_id = request.form['anaquel_id'] 

    if imagen_file.filename == '':
        return jsonify({"error": "No se seleccionó ningún archivo de imagen"}), 400
    
    _, ext = os.path.splitext(imagen_file.filename)
    if not ext.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']: ext = ".jpg" 
    filename = str(uuid.uuid4()) + ext 
    ruta_imagen_temporal = os.path.join(UPLOADS_DIR, filename)
    
    try:
        imagen_file.save(ruta_imagen_temporal)
        print(f"API: Imagen guardada temporalmente en: {ruta_imagen_temporal}")
        # PATH_MODELO_YOLO_PRINCIPAL se pasa aquí como la ruta absoluta al modelo.
        # code_processor.py lo usará.
        resultados = procesar_anaquel_para_api(
            ruta_imagen_temporal,
            anaquel_id,
            PATH_MODELO_YOLO_PRINCIPAL, 
            BASE_DIR 
        )
    except Exception as e:
        print(f"Error catastrófico durante el procesamiento en endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Error interno del servidor durante el análisis."}), 500
    finally:
        if os.path.exists(ruta_imagen_temporal):
            try: os.remove(ruta_imagen_temporal); print(f"API: Imagen temporal eliminada.")
            except Exception as e_del: print(f"Error al eliminar imagen temporal: {e_del}")
    
    if "error" in resultados:
         return jsonify(resultados), resultados.get("status_code", 400)
    return jsonify(resultados)

@app.route('/api/registrar-feedback', methods=['POST'])
def registrar_feedback_endpoint():
    try:
        data = request.get_json()
        if not data: return jsonify({"error": "No se recibieron datos de feedback"}), 400
        
        feedback_file_path = os.path.join(BASE_DIR, "feedback_log.txt")
        with open(feedback_file_path, "a", encoding="utf-8") as f:
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            for key, value in data.items():
                f.write(f"{key.replace('_', ' ').capitalize()}: {value}\n")
            f.write("-" * 20 + "\n")
        print(f"--- FEEDBACK RECIBIDO Y GUARDADO ---\n{data}")
        return jsonify({"mensaje": "Feedback recibido correctamente"}), 200
    except Exception as e:
        print(f"Error al procesar feedback: {e}")
        return jsonify({"error": f"Error interno del servidor al procesar feedback: {str(e)}"}), 500

if __name__ == '__main__':
    if not os.path.exists(PATH_MODELO_YOLO_PRINCIPAL):
        print(f"ERROR CRÍTICO: El modelo YOLO no se encuentra en: {PATH_MODELO_YOLO_PRINCIPAL}")
        print("La API no se iniciará. Por favor, copia tu modelo 'best(2).pt' a la ruta esperada o ajusta PATH_MODELO_YOLO_PRINCIPAL en api.py.")
    else:
        print(f"API Iniciada. Modelo YOLO principal en: {PATH_MODELO_YOLO_PRINCIPAL}.")
        print(f"Archivos de configuración de anaqueles (relativos a '{BASE_DIR}') serán cargados por code_processor.py.")
        app.run(debug=True, host='0.0.0.0', port=5000)