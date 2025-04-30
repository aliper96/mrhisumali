import h5py
import numpy as np
from tqdm import tqdm


def debug_last_batch():
    # List of videos from the last batch where the error occurs
    last_batch_videos = [
        'video_9952', 'video_9953', 'video_9954', 'video_9955', 'video_9956',
        'video_9957', 'video_9958', 'video_9959', 'video_996', 'video_9960',
        'video_9961', 'video_9962', 'video_9963', 'video_9964', 'video_9965',
        'video_9966', 'video_9967', 'video_9968', 'video_9969', 'video_997',
        'video_9970', 'video_9971', 'video_9972', 'video_9973', 'video_9974',
        'video_9975', 'video_9976', 'video_9977', 'video_9978', 'video_9979',
        'video_998', 'video_9980', 'video_9981', 'video_999'
    ]

    h5_file_path = "mr_hisum.h5"

    print("Analyzing dimensions of videos in the last batch...")
    with h5py.File(h5_file_path, 'r') as h5:
        for vid in tqdm(last_batch_videos):
            try:
                # Check if video exists in h5 file
                if vid not in h5:
                    print(f"{vid}: Not found in h5 file")
                    continue

                # Check available datasets for this video
                available_keys = list(h5[vid].keys())
                print(f"{vid}: Available keys: {available_keys}")

                # Check dimensions of each dataset
                for key in available_keys:
                    shape = h5[vid][key].shape
                    print(f"  - {key}: shape = {shape}")

                # Specifically check features and gtscore
                if 'features' in h5[vid] and 'gtscore' in h5[vid]:
                    features_shape = h5[vid]['features'].shape
                    gtscore_shape = h5[vid]['gtscore'].shape

                    if features_shape[0] != gtscore_shape[0]:
                        print(
                            f"⚠️ DIMENSION MISMATCH in {vid}: features={features_shape[0]}, gtscore={gtscore_shape[0]}")

                    # Also check gt_summary if it exists
                    if 'gt_summary' in h5[vid]:
                        gt_summary_shape = h5[vid]['gt_summary'].shape
                        if features_shape[0] != gt_summary_shape[0]:
                            print(
                                f"⚠️ DIMENSION MISMATCH in {vid}: features={features_shape[0]}, gt_summary={gt_summary_shape[0]}")

                        # Check specific values (first and last)
                        gt_summary = h5[vid]['gt_summary'][:]
                        print(f"  - gt_summary: first 5 values = {gt_summary[:5]}, last 5 values = {gt_summary[-5:]}")

                    # Check for change_points which might also be used in evaluation
                    if 'change_points' in h5[vid]:
                        change_points = h5[vid]['change_points'][:]
                        print(f"  - change_points: shape = {change_points.shape}, content = {change_points}")

            except Exception as e:
                print(f"Error analyzing {vid}: {str(e)}")


if __name__ == "__main__":
    debug_last_batch()