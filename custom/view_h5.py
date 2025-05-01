

import h5py

with h5py.File(r"C:\Users\aliha\Desktop\ABYYS\MR.HiSum\dataset\mr_hisum.h5", 'r') as f:
    if 'video_2' in f:
        dataset = f['video_2']  # SIN DOS PUNTOS (:)
        change_points = dataset['change_points'][:]
        features = dataset['features'][:]
        gt_summary = dataset['gt_summary'][:]
        gtscore = dataset['gtscore'][:]

        print("change_points:", change_points)
        # print("features:", features)
        print("gt_summary:", gt_summary)
        print("gtscore:", gtscore)

    else:
        print("No existe la clave 'video_2'.")