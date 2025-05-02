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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import threading
import queue

# Importar módulos de detección de cambios
try:
    from cpd_nonlin import cpd_nonlin
    from cpd_auto import cpd_auto
    KTS_AVAILABLE = True
except ImportError:
    print("ADVERTENCIA: Módulos KTS no disponibles; usando KMeans.")
    from sklearn.cluster import KMeans
    KTS_AVAILABLE = False

# Importar extractor de features y frame iterator
from custom.feature_extractor import YouTube8MFeatureExtractor
from custom.extract_tfrecords_main import frame_iterator

# Optimizar mochila con Numba
try:
    from numba import njit
    @njit
    def knapsack_dp(lengths, scores, W):
        M = lengths.shape[0]
        Kmat = np.zeros((M + 1, W + 1), np.float32)
        for i in range(1, M + 1):
            li, si = lengths[i-1], scores[i-1]
            for w in range(W + 1):
                if li <= w:
                    val = Kmat[i-1, w-li] + si
                    if val > Kmat[i-1, w]:
                        Kmat[i, w] = val
                    else:
                        Kmat[i, w] = Kmat[i-1, w]
                else:
                    Kmat[i, w] = Kmat[i-1, w]
        return Kmat
    USE_NUMBA = True
except ImportError:
    print("Numba no instalado; mochila en Python puro.")
    USE_NUMBA = False

# Variables globales para workers
global h5_lock, H5_PATH, MODEL_DIR, MAX_SECS, N_SEGMENTS
h5_lock = None
H5_PATH = None
MODEL_DIR = None
MAX_SECS = None
N_SEGMENTS = None


def init_worker(lock, h5_path, model_dir, max_secs, n_segments):
    global h5_lock, H5_PATH, MODEL_DIR, MAX_SECS, N_SEGMENTS
    h5_lock = lock
    H5_PATH = h5_path
    MODEL_DIR = model_dir
    MAX_SECS = max_secs
    N_SEGMENTS = n_segments


def extract_yt8m_features(video_path, model_dir, max_secs=300):
    extractor = YouTube8MFeatureExtractor(model_dir=model_dir)
    q = queue.Queue(maxsize=8)
    feats = []

    def reader():
        for frame in frame_iterator(video_path, every_ms=1000.0, max_num_frames=max_secs):
            q.put(frame)
        q.put(None)

    def worker():
        while True:
            frame = q.get()
            if frame is None:
                break
            rgb = frame[:, :, ::-1]
            vec = extractor.extract_rgb_frame_features(rgb, apply_pca=True)
            feats.append(vec.astype(np.float32))

    t_r = threading.Thread(target=reader)
    t_w = threading.Thread(target=worker)
    t_r.start(); t_w.start()
    t_r.join(); t_w.join()

    if not feats:
        raise ValueError(f"No frames extraídos de {video_path}")
    return np.stack(feats, axis=0)


def load_heatmap(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    pos  = np.array([d['position'] for d in data])
    heat = np.array([d['heat']     for d in data])
    return pos, heat


def compute_gtscore(pos, heat, duration):
    times = np.arange(duration) / duration
    return np.interp(times, pos, heat)


def compute_change_points(features, max_cps=50):
    K = features.dot(features.T)
    N = K.shape[0]
    vmax = np.trace(K) / float(N)
    if KTS_AVAILABLE:
        cps, _ = cpd_auto(K, ncp=max_cps, vmax=vmax, desc_rate=1)
    else:
        km = KMeans(n_clusters=max_cps).fit(K)
        cps = sorted(set([int(c) for c in km.labels_]))
    cps = np.unique(np.concatenate(([0], np.array(cps, dtype=int), [N])))
    cps.sort()
    return np.vstack([[cps[i], cps[i+1]] for i in range(len(cps)-1)])


def compute_gtsummary(gtscore, change_points, budget_ratio=0.40):
    lengths = change_points[:, 1] - change_points[:, 0]
    scores = np.array([gtscore[start:end].sum() for start,end in change_points], dtype=np.float32)
    W = int(budget_ratio * len(gtscore))
    M = len(lengths)
    if USE_NUMBA:
        Kmat = knapsack_dp(lengths, scores, W)
    else:
        Kmat = np.zeros((M+1, W+1), dtype=np.float32)
        for i in range(1, M+1):
            li, si = lengths[i-1], scores[i-1]
            for w in range(W+1):
                if li <= w:
                    Kmat[i, w] = max(Kmat[i-1, w], Kmat[i-1, w-li] + si)
                else:
                    Kmat[i, w] = Kmat[i-1, w]
    w = W
    selected = np.zeros(M, dtype=np.int8)
    for i in range(M, 0, -1):
        if Kmat[i, w] != Kmat[i-1, w]:
            selected[i-1] = 1
            w -= lengths[i-1]
    summary = np.zeros(len(gtscore), dtype=np.float32)
    for sel,(start,end) in zip(selected, change_points):
        if sel:
            summary[start:end] = 1.0
    return summary


def process_and_store(video_dir):
    dataset_dir = os.environ['DATASET_DIR']
    vp = Path(dataset_dir)/video_dir
    video_path = next((str(p) for ext in ['mp4','mkv','webm']
                       for p in vp.glob(f"video.{ext}")), None)
    heatmap = next(vp.glob("*.json"), None)
    if not video_path or not heatmap:
        return False, video_dir
    try:
        feats = extract_yt8m_features(video_path, MODEL_DIR, MAX_SECS)
        duration = feats.shape[0]
        pos, heat = load_heatmap(str(heatmap))
        gtscore = compute_gtscore(pos, heat, duration)
        cps = compute_change_points(feats, max_cps=N_SEGMENTS)
        gtsum = compute_gtsummary(gtscore, cps)
        with h5_lock:
            with h5py.File(H5_PATH, 'a') as hf:
                grp = hf.create_group(video_dir)
                grp.create_dataset('features', data=feats, compression='gzip')
                grp.create_dataset('gtscore', data=gtscore)
                grp.create_dataset('change_points', data=cps)
                grp.create_dataset('gt_summary', data=gtsum)
                grp.attrs['duration_seconds'] = duration
        return True, video_dir
    except Exception as e:
        print(f"Error en {video_dir}: {e}")
        return False, video_dir


def main():
    parser = argparse.ArgumentParser()
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

    os.environ['DATASET_DIR'] = args.dataset_dir
    manager = mp.Manager()
    lock = manager.Lock()

    video_dirs = [d for d in os.listdir(args.dataset_dir)
                  if (Path(args.dataset_dir)/d).is_dir()]
    print(os.cpu_count())

    # Inicializar pool
    with ProcessPoolExecutor(
        max_workers=8,
        initializer=init_worker,
        initargs=(lock, args.output_h5, args.model_dir, args.max_secs, args.n_segments)
    ) as exe:
        futures = {exe.submit(process_and_store, vd): vd for vd in video_dirs}
        processed = []
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Procesando videos"):
            ok, vd = fut.result()
            if ok:
                processed.append(vd)

    # Crear split JSON
    random.shuffle(processed)
    n = len(processed)
    n_train = int(0.7*n)
    n_val   = int(0.15*n)
    split = {
        'train': processed[:n_train],
        'val'  : processed[n_train:n_train+n_val],
        'test' : processed[n_train+n_val:]
    }
    with open(args.output_split, 'w') as f:
        json.dump(split, f, indent=2)

    print(f"Proceso completado. Videos: {len(processed)}")

if __name__ == '__main__':
    main()
