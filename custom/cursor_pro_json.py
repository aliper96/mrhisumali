#!/usr/bin/env python3
import os
import sys
import argparse
import json
import numpy as np
import h5py
import cv2
from pathlib import Path

# Importar módulos necesarios
try:
    from cpd_nonlin import cpd_nonlin
    from cpd_auto import cpd_auto
    KTS_AVAILABLE = True
except ImportError:
    print("ADVERTENCIA: No se pudo importar el módulo KTS (cpd_nonlin/cpd_auto).")
    print("Se usará una implementación alternativa basada en K-means.")
    from sklearn.cluster import KMeans
    KTS_AVAILABLE = False

# Importar extractor de features
from feature_extractor import YouTube8MFeatureExtractor
from extract_tfrecords_main import frame_iterator

def extract_yt8m_features(video_path, model_dir, max_secs=300):
    """
    Extrae features de Inception+PCA (1024-D) a 1 fps de un video local.
    - video_path: ruta al vídeo.
    - model_dir: carpeta donde se descargan los modelos (Inception .tgz + PCA .pb).
    - max_secs: máximos segundos a procesar (YouTube-8M original grafica 300s).
    Retorna un ndarray de shape (n_secs, 1024) dtype float32.
    """
    # Verificar que OpenCV puede abrir el vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el vídeo: {video_path}")
    cap.release()

    # Instanciar el extractor oficial (descarga modelos si falta)
    extractor = YouTube8MFeatureExtractor(model_dir=model_dir)

    feats = []
    # Iterar a 1 fps (every_ms=1000) hasta max_secs
    for frame in frame_iterator(video_path, every_ms=1000.0, max_num_frames=max_secs):
        # frame viene en BGR, convertir a RGB
        rgb = frame[:, :, ::-1]
        # Extraer 1024-D PCA feature
        vec = extractor.extract_rgb_frame_features(rgb, apply_pca=True)
        feats.append(vec.astype(np.float32))

    if len(feats) == 0:
        raise ValueError(f"No se extrajo ningún frame de {video_path}")

    return np.stack(feats, axis=0)

def load_heatmap(json_path):
    """
    Carga el JSON de heatmap y devuelve vectores de posiciones normalizadas y valores de heat.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    pos  = np.array([d['position'] for d in data])
    heat = np.array([d['heat']     for d in data])
    return pos, heat

def compute_gtscore(pos, heat, duration):
    """
    Interpola el heatmap (valores normalizados) a un vector de longitud = segundos del vídeo.
    """
    times = np.arange(duration) / duration
    return np.interp(times, pos, heat)


def compute_change_points(features, max_cps=50):
    """
    Detecta límites de shots usando cpd_auto sobre el kernel lineal de features,
    y devuelve una lista de pares [inicio, fin] para cada segmento.
    """
    # 1) Kernel lineal
    K = features.dot(features.T)
    N = K.shape[0]
    vmax = np.trace(K) / float(N)

    # 2) Llamada a cpd_auto
    cps, _ = cpd_auto(K, ncp=max_cps, vmax=vmax, desc_rate=1)
    cps = np.array(cps, dtype=int)

    # 3) Garantizar 0 y N
    cps = np.concatenate(([0], cps, [N]))
    cps = np.unique(cps)
    cps = np.sort(cps)

    # 4) Convertir a pares [inicio, fin]
    segments = []
    for i in range(len(cps) - 1):
        segments.append([cps[i], cps[i + 1]])

    return np.array(segments)


def compute_gtsummary(gtscore, change_points, budget_ratio=0.15):
    """
    Resuelve la mochila 0/1 para seleccionar shots bajo un presupuesto de tiempo.
    Devuelve un vector binario por segundo indicando resumen.
    """
    lengths = change_points[:, 1] - change_points[:, 0]  # Calcular longitud de cada segmento
    scores = [gtscore[start:end].sum() for start, end in change_points]
    W = int(budget_ratio * len(gtscore))
    M = len(lengths)

    # Matriz DP
    Kmat = np.zeros((M + 1, W + 1))
    for i in range(1, M + 1):
        for w in range(W + 1):
            if lengths[i - 1] <= w:
                Kmat[i, w] = max(Kmat[i - 1, w],
                                 Kmat[i - 1, w - lengths[i - 1]] + scores[i - 1])
            else:
                Kmat[i, w] = Kmat[i - 1, w]

    # Reconstruir selección
    w = W
    selected = np.zeros(M, dtype=int)
    for i in range(M, 0, -1):
        if Kmat[i, w] != Kmat[i - 1, w]:
            selected[i - 1] = 1
            w -= lengths[i - 1]

    # Expandir a nivel de segundo
    summary = np.zeros(len(gtscore), dtype=float)  # Cambiar a float como en el ejemplo deseado
    for i, sel in enumerate(selected):
        if sel:
            start, end = change_points[i]
            summary[start:end] = 1.0

    return summary
def create_hisum_h5(video_path, heatmap_json_path, video_id, output_path, model_dir, max_secs=300, n_segments=20):
    """
    Crea un archivo H5 con todos los campos requeridos para Mr.HiSum
    
    Args:
        video_path: Ruta al archivo de video
        heatmap_json_path: Ruta al archivo JSON con el heatmap
        video_id: Identificador del video en el archivo H5
        output_path: Ruta donde guardar el archivo H5
        model_dir: Directorio donde están los modelos de YouTube-8M
        max_secs: Máximo número de segundos a procesar
        n_segments: Número máximo de segmentos a detectar
    """
    # 1. Extraer features del video
    print(f"Extrayendo features de {video_path}...")
    features = extract_yt8m_features(video_path, model_dir, max_secs)
    duration = features.shape[0]  # Duración en segundos
    
    # 2. Procesar heatmap y generar gtscore
    print(f"Procesando heatmap de {heatmap_json_path}...")
    pos, heat = load_heatmap(heatmap_json_path)
    gtscore = compute_gtscore(pos, heat, duration)
    
    # 3. Calcular change points
    print("Calculando change points...")
    change_points = compute_change_points(features, max_cps=n_segments)
    
    # 4. Generar resumen ground truth
    print("Generando resumen ground truth...")
    gtsummary = compute_gtsummary(gtscore, change_points)

    print(features)
    print(gtscore)
    print(change_points)
    print(gtsummary)
    
    # 5. Guardar todo en el archivo H5
    print(f"Guardando datos en {output_path}...")
    with h5py.File(output_path, 'w') as f:
        # Crear grupo para el video
        video_group = f.create_group(video_id)
        
        # Agregar datasets al grupo del video
        video_group.create_dataset('features', data=features, compression="gzip")
        video_group.create_dataset('gtscore', data=gtscore)
        video_group.create_dataset('change_points', data=change_points)
        video_group.create_dataset('gt_summary', data=gtsummary)
        
        # Guardar duración como atributo
        video_group.attrs['duration_seconds'] = duration
    
    print(f"✓ Proceso completado. Archivo {output_path} creado con éxito.")
    print(f"Contiene el grupo '{video_id}' con los campos: features, gtscore, change_points, gt_summary")

def main():
    parser = argparse.ArgumentParser(
        description="Genera archivo H5 para Mr.HiSum a partir de video y heatmap"
    )
    parser.add_argument("--video", default="C:/Users/aliha/Documents/videoplayback.mp4", help="Ruta al archivo de video")
    parser.add_argument("--heatmap", default=r"C:\Users\aliha\Documents\wq7rSbQx2G8.json", help="Ruta al archivo JSON de heatmap")
    parser.add_argument("--video_id", default="video2", help="ID del video en el archivo H5")
    parser.add_argument("--output", default="mr_hisum.h5", help="Archivo H5 de salida")
    parser.add_argument("--model_dir", default="inception_weights/", 
                       help="Directorio con modelos de YouTube-8M")
    parser.add_argument("--max_secs", type=int, default=300,
                       help="Segundos máximos por video (default 300)")
    parser.add_argument("--n_segments", type=int, default=20,
                       help="Número máximo de segmentos a detectar")
    
    args = parser.parse_args()
    
    create_hisum_h5(
        video_path=args.video,
        heatmap_json_path=args.heatmap,
        video_id=args.video_id,
        output_path=args.output,
        model_dir=args.model_dir,
        max_secs=args.max_secs,
        n_segments=args.n_segments
    )

if __name__ == "__main__":
    main() 