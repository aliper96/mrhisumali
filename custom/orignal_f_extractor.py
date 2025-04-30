#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import h5py
import cv2

# Asegúrate de que estos módulos estén en tu PYTHONPATH
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

def validate_feats(array, vid_id):
    """Chequeos básicos de sanity para los features."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"[{vid_id}] Features no es ndarray, es {type(array)}")
    if array.ndim != 2 or array.shape[1] != 1024:
        raise ValueError(f"[{vid_id}] Shape inesperado: {array.shape}")
    if not np.isfinite(array).all():
        cnt = np.logical_or(np.isnan(array), np.isinf(array)).sum()
        raise ValueError(f"[{vid_id}] Encontrados {cnt} valores no finitos")
    return True

def main():
    p = argparse.ArgumentParser(
        description="Extrae YouTube-8M frame-level features con Inception+PCA → H5"
    )
    p.add_argument("--video_list", default="test_video.csv",
                   help="CSV: video_id,video_path  (sin cabecera)")
    p.add_argument("--model_dir", default="inception_weights/",
                   help="Donde descargar o encontrar inception+PCA")
    p.add_argument("--output_h5", default="features_local.h5",
                   help="Archivo H5 de salida")
    p.add_argument("--max_secs", type=int, default=300,
                   help="Segundos máximos por video (default 300)")
    args = p.parse_args()

    # Leer lista de vídeos
    videos = []
    with open(args.video_list, 'r') as f:
        for line in f:
            vid, path = line.strip().split(",", 1)
            videos.append((vid, path))

    with h5py.File(args.output_h5, 'a') as h5f:
        for vid, path in videos:
            try:
                feats = extract_yt8m_features(path, args.model_dir, args.max_secs)
                validate_feats(feats, vid)
            except Exception as e:
                print(f"[ERROR] {vid}: {e}", file=sys.stderr)
                continue

            # Guardar/reemplazar grupo
            if vid in h5f:
                del h5f[vid]
            grp = h5f.create_group(vid)
            grp.create_dataset("features", data=feats, compression="gzip")
            grp.attrs["n_steps"] = feats.shape[0]

            print(f"✓ {vid}: extraídos {feats.shape[0]}s × 1024 dims")

    print(f"\nProceso completado. Salida en {args.output_h5}")

if __name__ == "__main__":
    main()
