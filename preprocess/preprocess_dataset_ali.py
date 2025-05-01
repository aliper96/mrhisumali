#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import h5py
import cv2
from pathlib import Path
import random
from tqdm import tqdm

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
from custom.feature_extractor import YouTube8MFeatureExtractor
from custom.extract_tfrecords_main import frame_iterator

def extract_yt8m_features(video_path, model_dir, max_secs=300):
    """
    Extrae features de Inception+PCA (1024-D) a 1 fps de un video local.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el vídeo: {video_path}")
    cap.release()

    extractor = YouTube8MFeatureExtractor(model_dir=model_dir)
    feats = []
    
    for frame in frame_iterator(video_path, every_ms=1000.0, max_num_frames=max_secs):
        rgb = frame[:, :, ::-1]
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
    Detecta límites de shots usando cpd_auto sobre el kernel lineal de features.
    """
    K = features.dot(features.T)
    N = K.shape[0]
    vmax = np.trace(K) / float(N)

    cps, _ = cpd_auto(K, ncp=max_cps, vmax=vmax, desc_rate=1)
    cps = np.array(cps, dtype=int)

    cps = np.concatenate(([0], cps, [N]))
    cps = np.unique(cps)
    cps = np.sort(cps)

    return cps

def compute_gtsummary(gtscore, change_points, budget_ratio=0.15):
    """
    Resuelve la mochila 0/1 para seleccionar shots bajo un presupuesto de tiempo.
    """
    lengths = np.diff(change_points)
    scores  = [gtscore[change_points[i]:change_points[i+1]].sum()
               for i in range(len(lengths))]
    W       = int(budget_ratio * len(gtscore))
    M       = len(lengths)
    
    Kmat = np.zeros((M+1, W+1))
    for i in range(1, M+1):
        for w in range(W+1):
            if lengths[i-1] <= w:
                Kmat[i,w] = max(Kmat[i-1,w],
                                Kmat[i-1,w-lengths[i-1]] + scores[i-1])
            else:
                Kmat[i,w] = Kmat[i-1,w]
    
    w        = W
    selected = np.zeros(M, dtype=int)
    for i in range(M, 0, -1):
        if Kmat[i,w] != Kmat[i-1,w]:
            selected[i-1] = 1
            w           -= lengths[i-1]
    
    summary = np.zeros(len(gtscore), dtype=int)
    for i, sel in enumerate(selected):
        if sel:
            s, e = change_points[i], change_points[i+1]
            summary[s:e] = 1
    return summary

def process_video(video_path, heatmap_path, video_id, h5_file, model_dir, max_secs=300, n_segments=20):
    """
    Procesa un video y su heatmap, guardando los resultados en el archivo H5.
    """
    try:
        # 1. Extraer features del video
        features = extract_yt8m_features(video_path, model_dir, max_secs)
        duration = features.shape[0]
        
        # 2. Procesar heatmap y generar gtscore
        pos, heat = load_heatmap(heatmap_path)
        gtscore = compute_gtscore(pos, heat, duration)
        
        # 3. Calcular change points
        change_points = compute_change_points(features, max_cps=n_segments)
        
        # 4. Generar resumen ground truth
        gtsummary = compute_gtsummary(gtscore, change_points)
        
        # 5. Guardar en el archivo H5
        video_group = h5_file.create_group(video_id)
        video_group.create_dataset('features', data=features, compression="gzip")
        video_group.create_dataset('gtscore', data=gtscore)
        video_group.create_dataset('change_points', data=change_points)
        video_group.create_dataset('gt_summary', data=gtsummary)
        video_group.attrs['duration_seconds'] = duration
        
        return True
        
    except Exception as e:
        print(f"Error procesando {video_id}: {str(e)}")
        return False

def create_split_json(video_ids, output_path, train_ratio=0.7, val_ratio=0.15):
    """
    Crea un archivo JSON de split similar al original.
    """
    random.shuffle(video_ids)
    n_total = len(video_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    split = {
        "train": video_ids[:n_train],
        "val": video_ids[n_train:n_train+n_val],
        "test": video_ids[n_train+n_val:]
    }
    
    with open(output_path, 'w') as f:
        json.dump(split, f, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description="Procesa múltiples videos y heatmaps para Mr.HiSum"
    )
    parser.add_argument("--dataset_dir", default=r"D:\dataset_abyss\filtered_videos_1433",
                       help="Directorio raíz con carpetas de videos")
    parser.add_argument("--model_dir", default="custom/inception_weights/",
                       help="Directorio con modelos de YouTube-8M")
    parser.add_argument("--output_h5", default="mr_ali.h5", 
                       help="Archivo H5 de salida")
    parser.add_argument("--output_split", default="mr_ali_split.json", 
                       help="Archivo JSON de split")
    parser.add_argument("--max_secs", type=int, default=300,
                       help="Segundos máximos por video")
    parser.add_argument("--n_segments", type=int, default=20,
                       help="Número máximo de segmentos")
    
    args = parser.parse_args()
    
    # Crear archivo H5
    h5_file = h5py.File(args.output_h5, 'w')
    video_ids = []
    
    # Encontrar todas las carpetas de videos
    video_dirs = [d for d in os.listdir(args.dataset_dir) 
                 if os.path.isdir(os.path.join(args.dataset_dir, d))]
    
    # Procesar cada video
    for video_dir in tqdm(video_dirs, desc="Procesando videos"):
        video_path = None
        heatmap_path = None
        
        # Buscar archivo de video
        for ext in ['.mp4', '.mkv', '.webm']:
            path = os.path.join(args.dataset_dir, video_dir, f'video{ext}')
            if os.path.exists(path):
                video_path = path
                break
        
        # Buscar archivo JSON de heatmap
        json_files = [f for f in os.listdir(os.path.join(args.dataset_dir, video_dir)) 
                     if f.endswith('.json')]
        if json_files:
            heatmap_path = os.path.join(args.dataset_dir, video_dir, json_files[0])
        
        if video_path and heatmap_path:
            if process_video(video_path, heatmap_path, video_dir, h5_file, 
                           args.model_dir, args.max_secs, args.n_segments):
                video_ids.append(video_dir)
    
    h5_file.close()
    
    # Crear archivo de split
    create_split_json(video_ids, args.output_split)
    
    print(f"\nProceso completado:")
    print(f"- Archivo H5: {args.output_h5}")
    print(f"- Archivo split: {args.output_split}")
    print(f"- Videos procesados: {len(video_ids)}")

if __name__ == "__main__":
    main() 