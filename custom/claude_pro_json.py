import json
import numpy as np
import h5py
import os

# Suponiendo que cpd_nonlin.pyx ha sido compilado como una extensión C
# y que cpd_auto.py está disponible en el mismo directorio
try:
    from cpd_nonlin import cpd_nonlin
    from cpd_auto import cpd_auto

    KTS_AVAILABLE = True
except ImportError:
    print("ADVERTENCIA: No se pudo importar el módulo KTS (cpd_nonlin/cpd_auto).")
    print("Se usará una implementación alternativa basada en K-means.")
    from sklearn.cluster import KMeans

    KTS_AVAILABLE = False


def process_heatmap_to_gtscore(heatmap_json, video_duration_seconds=None):
    """
    Convierte el JSON de heatmap a un array numpy para gtscore
    Si se proporciona video_duration_seconds, convierte las posiciones relativas a tiempo en segundos
    """
    # Extraer posiciones y valores de calor del JSON
    positions = np.array([item['position'] for item in heatmap_json])
    heat_values = np.array([item['heat'] for item in heatmap_json])

    # Asegurarse de que los valores están ordenados por posición
    sort_idx = np.argsort(positions)
    positions = positions[sort_idx]
    heat_values = heat_values[sort_idx]

    # Si se proporciona la duración del video, convertir posiciones a segundos
    if video_duration_seconds is not None:
        positions_seconds = positions * video_duration_seconds
    else:
        positions_seconds = positions

    return positions_seconds, heat_values


def create_kernel_matrix(features):
    """
    Crea una matriz de kernel a partir de características
    Para el algoritmo KTS, se necesita una matriz de kernel
    que mida la similitud entre frames
    """
    n = len(features)
    K = np.zeros((n, n))

    # Calculamos la matriz de kernel como el producto escalar de las características
    for i in range(n):
        for j in range(i, n):
            # Para el heatmap, podemos usar una medida de similitud basada en la diferencia
            # entre valores de heat adyacentes
            diff = abs(features[i] - features[j])
            similarity = np.exp(-diff)  # Kernel RBF sencillo
            K[i, j] = similarity
            K[j, i] = similarity  # La matriz debe ser simétrica

    return K


def detect_change_points_kts(heat_values, n_segments=10):
    """
    Detecta puntos de cambio utilizando el algoritmo Kernel Temporal Segmentation (KTS)
    """
    # Crear matriz de kernel
    K = create_kernel_matrix(heat_values)

    # Parámetros para KTS
    max_cp = n_segments  # Máximo número de puntos de cambio
    vmax = 1.0  # Parámetro especial (puedes ajustarlo)
    lmin = max(len(heat_values) // (max_cp * 10), 2)  # Longitud mínima del segmento

    # Ejecutar KTS
    cps, _ = cpd_auto(K, max_cp, vmax, lmin=lmin)

    # Crear segmentos con los puntos de cambio
    segments = []
    start_idx = 0

    # Añadir todos los puntos de cambio como límites de segmentos
    for cp in sorted(cps):
        segments.append([start_idx, cp])
        start_idx = cp

    # Añadir el último segmento
    segments.append([start_idx, len(heat_values)])

    return np.array(segments)


def detect_change_points_kmeans(heat_values, n_segments=10):
    """
    Detecta puntos de cambio utilizando KMeans como alternativa a KTS
    """
    # Crear características para segmentación (posición y calor)
    X = np.column_stack((np.arange(len(heat_values)), heat_values))

    # Aplicar K-means para encontrar segmentos
    kmeans = KMeans(n_clusters=n_segments, random_state=0, n_init=10).fit(X)
    labels = kmeans.labels_

    # Encontrar límites donde cambian los labels
    change_indices = [0]  # Empezamos con el índice 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            change_indices.append(i)
    change_indices.append(len(labels))  # Añadimos el último índice

    # Crear los pares [inicio, fin] para cada segmento
    segments = []
    for i in range(len(change_indices) - 1):
        start = change_indices[i]
        end = change_indices[i + 1]
        segments.append([start, end])

    return np.array(segments)


def detect_change_points(heat_values, n_segments=10):
    """
    Función principal para detectar puntos de cambio.
    Usa KTS si está disponible, de lo contrario usa K-means.
    """
    if KTS_AVAILABLE:
        return detect_change_points_kts(heat_values, n_segments)
    else:
        return detect_change_points_kmeans(heat_values, n_segments)


def solve_knapsack(heat_values, segments, max_duration=0.3):
    """
    Implementa un algoritmo de mochila 0/1 simple para seleccionar los mejores segmentos
    basados en su popularidad (heat) para crear un resumen.
    """
    if len(segments) == 0:
        return np.zeros(len(heat_values))

    # Calcular valor promedio de heat para cada segmento
    segment_values = []
    segment_durations = []

    for i in range(len(segments)):
        start, end = segments[i]
        if start < len(heat_values) and end <= len(heat_values):
            segment_heat = np.mean(heat_values[start:end])
            segment_duration = (end - start) / len(heat_values)  # Duración normalizada

            segment_values.append(segment_heat)
            segment_durations.append(segment_duration)

    # Algoritmo de la mochila 0/1
    n = len(segment_values)
    # Convertir max_duration a unidades discretas (porcentaje de la duración total)
    max_capacity = int(max_duration * 100)  # Multiplicar por 100 para tener precisión
    # Convertir duraciones a unidades discretas
    discrete_durations = [int(d * 100) for d in segment_durations]

    # Tabla de programación dinámica
    dp = [[0 for _ in range(max_capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(max_capacity + 1):
            if discrete_durations[i - 1] <= w:
                dp[i][w] = max(
                    segment_values[i - 1] + dp[i - 1][w - discrete_durations[i - 1]],
                    dp[i - 1][w]
                )
            else:
                dp[i][w] = dp[i - 1][w]

    # Reconstruir la solución
    selected_segments = []
    w = max_capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_segments.append(i - 1)
            w -= discrete_durations[i - 1]

    # Convertir a array binario: 1 para los frames seleccionados, 0 para los no seleccionados
    binary_summary = np.zeros(len(heat_values))

    # Marcar como 1 todas las posiciones dentro de los segmentos seleccionados
    for seg_idx in selected_segments:
        if seg_idx < len(segments):
            start, end = segments[seg_idx]
            for pos in range(start, end):
                if 0 <= pos < len(binary_summary):
                    binary_summary[pos] = 1.0

    return binary_summary


def create_hisum_h5(heatmap_json, video_id='video_1', output_path='mr_hisum.h5',
                    n_segments=20, video_duration_seconds=None):
    """
    Crea un archivo H5 con los campos requeridos para Mr.HiSum

    Parámetros:
    - heatmap_json: JSON con datos de popularidad del video
    - video_id: Identificador del video en el archivo H5
    - output_path: Ruta donde guardar el archivo H5
    - n_segments: Número máximo de segmentos a detectar
    - video_duration_seconds: Duración del video en segundos (opcional)
    """
    positions, gtscore = process_heatmap_to_gtscore(heatmap_json, video_duration_seconds)

    # Detectar puntos de cambio usando KTS o K-means
    change_points = detect_change_points(gtscore, n_segments)

    # Generar resumen ground truth
    gtsummary = solve_knapsack(gtscore, change_points)

    # Mostrar los datos generados para verificación
    print("gtscore shape:", gtscore.shape)
    print("change_points shape:", change_points.shape)
    print("gtsummary shape:", gtsummary.shape)
    print(gtscore)
    print(change_points)
    print(gtsummary)

    # Crear archivo H5 con estructura correcta
    with h5py.File(output_path, 'w') as f:
        # Crear grupo para el video
        video_group = f.create_group(video_id)

        # Agregar datasets al grupo del video
        video_group.create_dataset('gtscore', data=gtscore)
        video_group.create_dataset('change_points', data=change_points)
        video_group.create_dataset('gt_summary', data=gtsummary)  # Nombre correcto: gt_summary, no gtsummary

        # Si se proporciona la duración del video, guardarla como atributo
        if video_duration_seconds is not None:
            video_group.attrs['duration_seconds'] = video_duration_seconds

    print(f"Archivo {output_path} creado con éxito.")
    print(f"Contiene el grupo '{video_id}' con los campos: gtscore, change_points, gt_summary")
    print(f"Usa el script de preprocesamiento para agregar los features.")


# Ejemplo de uso
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generar archivo H5 para Mr.HiSum a partir de JSON de heatmap')
    parser.add_argument('--json_file', type=str, default= r"C:\Users\aliha\Documents\wq7rSbQx2G8.json", help='Ruta al archivo JSON de heatmap')
    parser.add_argument('--output', type=str, default='mr_hisum.h5', help='Ruta de salida para el archivo H5')
    parser.add_argument('--video_id', type=str, default='video_1', help='ID del video en el archivo H5')
    parser.add_argument('--segments', type=int, default=20, help='Número máximo de segmentos a detectar')
    parser.add_argument('--duration', type=float,default=136, help='Duración del video en segundos (opcional)')

    args = parser.parse_args()

    if os.path.exists(args.json_file):
        with open(args.json_file, 'r') as f:
            heatmap_data = json.load(f)

        create_hisum_h5(
            heatmap_data,
            video_id=args.video_id,
            output_path=args.output,
            n_segments=args.segments,
            video_duration_seconds=args.duration
        )
    else:
        print(f"El archivo {args.json_file} no existe.")
        print("Debes guardar tu JSON de heatmap en un archivo primero.")