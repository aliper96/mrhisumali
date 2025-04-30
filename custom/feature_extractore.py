import cv2
import mediapipe as mp
import numpy as np
import h5py
import json
import argparse
import sys


def extract_pose_landmarks(frame, pose_model):
    """
    Procesa un frame con MediaPipe Pose y devuelve los landmarks de pose.
    Devuelve una lista de floats con [x, y, z, visibility] por cada punto (33 puntos de pose).
    Si no se detecta pose en el frame, devuelve una lista de ceros de longitud 132.
    """
    # Convertir el frame de BGR (OpenCV) a RGB (MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Procesar el frame con el modelo de pose
    results = pose_model.process(frame_rgb)
    if results.pose_landmarks:
        # Si se detectaron landmarks, extraer x, y, z, visibility de cada uno
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        return landmarks
    else:
        # Si no se detectó pose, retornar 132 valores 0.0
        return [0.0] * (33 * 4)


def align_heatmap_to_frames(heatmap_data, num_frames):
    """
    Alinea los valores del heatmap normalizado a cada frame del video mediante interpolación.
    - heatmap_data: lista de puntos (x,y) donde x es tiempo normalizado (0-1) y y es valor normalizado (0-1).
                    También puede ser un diccionario que contiene dicha lista.
    - num_frames: número total de frames del video.
    Retorna un arreglo numpy 'gtscore' de longitud num_frames con un valor por frame.
    """
    # Extraer listas de tiempos (x) y valores (y) del heatmap_data
    times = []
    values = []

    # Si es un diccionario, intentamos encontrar la lista de puntos dentro de él
    if isinstance(heatmap_data, dict):
        # Buscamos si hay alguna clave principal que contenga los datos
        heatmap_points = None

        # Verificar si hay una clave específica que podría contener los datos
        possible_keys = ['heatmap', 'points', 'data', 'values']
        for key in possible_keys:
            if key in heatmap_data and isinstance(heatmap_data[key], list):
                heatmap_points = heatmap_data[key]
                break

        # Si no encontramos una clave específica, intentamos encontrar cualquier lista
        if heatmap_points is None:
            for key, val in heatmap_data.items():
                if isinstance(val, list) and len(val) > 0:
                    heatmap_points = val
                    break

        # Si encontramos una lista de puntos, usamos esa
        if heatmap_points is not None:
            heatmap_data = heatmap_points
        else:
            # No pudimos encontrar una lista de puntos, retornar ceros
            print("Warning: Could not find heatmap points in the JSON data")
            return np.zeros(num_frames, dtype=np.float32)

    # Ahora trabajamos con la lista de puntos (que puede ser la original o extraída del diccionario)
    if isinstance(heatmap_data, list):
        if len(heatmap_data) == 0:
            # Lista vacía: retornar gtscore de ceros
            return np.zeros(num_frames, dtype=np.float32)

        # Determinar formato de los puntos en la lista
        first = heatmap_data[0]

        if isinstance(first, dict):
            # Listas específicas de posibles nombres de claves
            possible_x_keys = ['x', 'time', 't', 'position']
            possible_y_keys = ['y', 'value', 'v', 'heat']

            # Intentar encontrar claves para x e y
            x_key = None
            y_key = None

            # Comprobar cada clave posible
            for key in possible_x_keys:
                if key in first:
                    x_key = key
                    break

            for key in possible_y_keys:
                if key in first:
                    y_key = key
                    break

            if x_key and y_key:
                print(f"Found keys: x={x_key}, y={y_key}")
                times = [float(point[x_key]) for point in heatmap_data]
                values = [float(point[y_key]) for point in heatmap_data]
            else:
                # No pudimos encontrar claves adecuadas, imprimir las claves disponibles y retornar ceros
                print(
                    f"Warning: Could not identify time/value keys in heatmap dictionaries. Available keys: {list(first.keys())}")
                return np.zeros(num_frames, dtype=np.float32)

        elif (isinstance(first, list) or isinstance(first, tuple)) and len(first) >= 2:
            # Formato lista/tupla de pares [x, y]
            times = [float(point[0]) for point in heatmap_data]
            values = [float(point[1]) for point in heatmap_data]

        elif isinstance(first, (int, float, str)):
            # Si es lista de valores simples, asumimos que es lista de valores uniformemente distribuidos
            try:
                values = [float(v) for v in heatmap_data]
                times = np.linspace(0.0, 1.0, num=len(values)).tolist()
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert heatmap values to float: {e}")
                return np.zeros(num_frames, dtype=np.float32)
        else:
            # Formato desconocido, imprimir información de debugging
            print(f"Warning: Unrecognized heatmap data format. First element type: {type(first)}")
            return np.zeros(num_frames, dtype=np.float32)
    else:
        # Formato inesperado, retornar ceros
        print(f"Warning: Unexpected heatmap_data type: {type(heatmap_data)}")
        return np.zeros(num_frames, dtype=np.float32)

    # Asegurar que los tiempos estén ordenados de forma ascendente
    sorted_pairs = sorted(zip(times, values), key=lambda x: x[0])
    times = [p[0] for p in sorted_pairs]
    values = [p[1] for p in sorted_pairs]

    # Extender el rango del heatmap a [0.0, 1.0] si es necesario
    if times[0] > 0.0:
        times.insert(0, 0.0)
        values.insert(0, 0.0)
    if times[-1] < 1.0:
        times.append(1.0)
        values.append(0.0)

    # Generar tiempo normalizado para cada frame
    frame_indices = np.arange(num_frames)
    if num_frames > 1:
        frame_times = frame_indices.astype(np.float64) / float(num_frames - 1)
    else:
        # Si solo hay un frame, asignar tiempo normalizado 0
        frame_times = np.array([0.0], dtype=np.float64)

    # Interpolar los valores del heatmap en los tiempos correspondientes a cada frame
    gtscore = np.interp(frame_times, times, values)
    return gtscore.astype(np.float32)
def main():
    # Configurar y parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description="Extraer características de pose de un video y alinear un heatmap a sus frames, guardando el resultado en un archivo H5.")
    parser.add_argument("--video_path", default=r"C:\Users\aliha\Documents\videoplayback.mp4", help="Ruta al video de entrada.")
    parser.add_argument("--heatmap_json", default=r"C:\Users\aliha\Documents\wq7rSbQx2G8.json", help="Ruta al archivo JSON del heatmap normalizado.")
    parser.add_argument("--output_h5", default="output_h5",
                        help="Ruta al archivo de salida .h5 donde guardar los resultados.")
    parser.add_argument("--video_id", default="test", help="Identificador del video dentro del archivo H5.")
    args = parser.parse_args()

    # Intentar abrir el video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en {args.video_path}", file=sys.stderr)
        sys.exit(1)

    # Leer el JSON de heatmap
    try:
        with open(args.heatmap_json, 'r') as f:
            heatmap_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Archivo JSON no encontrado en {args.heatmap_json}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: No se pudo decodificar el JSON de heatmap. Detalles: {e}", file=sys.stderr)
        sys.exit(1)

    # Inicializar el modelo de pose de MediaPipe
    try:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except Exception as e:
        print(f"Error: Falló la inicialización de MediaPipe Pose. Detalles: {e}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    features = []
    frame_count = 0

    try:
        # Procesar el video frame por frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Fin del video (o no se pudo leer más frames)
            frame_count += 1
            try:
                landmarks = extract_pose_landmarks(frame, pose)
            except Exception as e:
                # Si ocurre un error procesando este frame con MediaPipe, reportar aviso y continuar con frame vacío
                print(f"Advertencia: Falló el procesamiento de pose en el frame {frame_count}. Detalles: {e}",
                      file=sys.stderr)
                landmarks = [0.0] * (33 * 4)
            features.append(landmarks)
    finally:
        # Liberar los recursos de video y MediaPipe
        cap.release()
        pose.close()

    # Verificar si se procesó al menos un frame
    if frame_count == 0:
        print("Error: El video no contiene frames o no se pudo leer ningún frame.", file=sys.stderr)
        sys.exit(1)

    # Convertir la lista de features a un arreglo numpy
    features_array = np.array(features, dtype=np.float32)
    # Obtener vector de puntajes (gtscore) alineado a frames
    gtscore_array = align_heatmap_to_frames(heatmap_data, frame_count)

    print("Salida: ", gtscore_array)
    print("features", features_array)

    # Guardar en el archivo H5
    try:
        with h5py.File(args.output_h5, 'a') as h5f:
            # Si el grupo (video_id) ya existe, eliminarlo para actualizar sus datos
            if args.video_id in h5f:
                del h5f[args.video_id]
            grp = h5f.create_group(args.video_id)
            grp.create_dataset('features', data=features_array)
            grp.create_dataset('gtscore', data=gtscore_array)
    except Exception as e:
        print(f"Error: No se pudo guardar los resultados en {args.output_h5}. Detalles: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
