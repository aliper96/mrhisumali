import json
from pathlib import Path

import numpy as np
from cpd_auto import cpd_auto  # Módulo local para detección de change points


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
    y se asegura de que siempre haya 0 y N en el array resultante.
    """
    # 1) Kernel lineal
    K = features.dot(features.T)
    N = K.shape[0]
    vmax = np.trace(K) / float(N)

    # 2) Llamada a cpd_auto
    cps, _ = cpd_auto(K, ncp=max_cps, vmax=vmax, desc_rate=1)
    cps = np.array(cps, dtype=int)

    # 3) Garantizar 0 y N
    # Concatenamos siempre 0 y N, luego eliminamos duplicados y ordenamos
    cps = np.concatenate(([0], cps, [N]))
    cps = np.unique(cps)
    cps = np.sort(cps)

    return cps



def compute_gtsummary(gtscore, change_points, budget_ratio=0.15):
    """
    Resuelve la mochila 0/1 para seleccionar shots bajo un presupuesto de tiempo.
    Devuelve un vector binario por segundo indicando resumen.
    """
    lengths = np.diff(change_points)
    scores  = [gtscore[change_points[i]:change_points[i+1]].sum()
               for i in range(len(lengths))]
    W       = int(budget_ratio * len(gtscore))
    M       = len(lengths)
    # Matriz DP
    Kmat = np.zeros((M+1, W+1))
    for i in range(1, M+1):
        for w in range(W+1):
            if lengths[i-1] <= w:
                Kmat[i,w] = max(Kmat[i-1,w],
                                Kmat[i-1,w-lengths[i-1]] + scores[i-1])
            else:
                Kmat[i,w] = Kmat[i-1,w]
    # Reconstruir selección
    w        = W
    selected = np.zeros(M, dtype=int)
    for i in range(M, 0, -1):
        if Kmat[i,w] != Kmat[i-1,w]:
            selected[i-1] = 1
            w           -= lengths[i-1]
    # Expandir a nivel de segundo
    summary = np.zeros(len(gtscore), dtype=int)
    for i, sel in enumerate(selected):
        if sel:
            s, e = change_points[i], change_points[i+1]
            summary[s:e] = 1
    return summary

# Ejemplo de uso:
#
# from pathlib import Path
json_path = Path(r"C:\Users\aliha\Documents\wq7rSbQx2G8.json")
duration  = 136  # segundos del vídeo
# features  = np.load("dataset/features_video_2.npy")  # array (duration, feature_dim)

features = np.random.rand(duration,1025)
#
pos, heat     = load_heatmap(json_path)
gtscore       = compute_gtscore(pos, heat, duration)
change_points = compute_change_points(features, max_cps=50)
summary       = compute_gtsummary(gtscore, change_points, budget_ratio=0.15)

print(gtscore)
print(change_points)
print(summary)

print()
#
# Ahora tienes gtscore, change_points y summary en memoria.
