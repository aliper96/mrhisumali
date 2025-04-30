import os
import json
import math
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def align_most_replayed(file_name, random_id, youtube_id, duration):
    mostreplayed_json_path = f"mostReplayed/{file_name}/{file_name}_{random_id}_{youtube_id}.json"

    aligned = []
    with open(mostreplayed_json_path, 'r') as fd:
        data = json.load(fd)
        mr_chunk_size = data['heatMarkers'][0]["heatMarkerRenderer"]["markerDurationMillis"]
        for n in range(int(duration)):
            bin_number = math.floor(n * 1000 / mr_chunk_size)
            if bin_number > 99.01:
                bin_number = 99

            aligned.append(data['heatMarkers'][bin_number]["heatMarkerRenderer"]["heatMarkerIntensityScoreNormalized"])

    return np.array(aligned)


def preprocess():
    meta_data = "metadata.csv"
    h5_file_path = "mr_hisum.h5"

    # Load the metadata
    df = pd.read_csv(meta_data)

    # Open the existing h5 file in read mode first to check what's already there
    print("Checking existing features and gtscore dimensions in the h5 file...")
    with h5py.File(h5_file_path, 'r') as h5check:
        existing_videos = list(h5check.keys())
        videos_to_keep = []
        videos_to_remove = []
        dimension_issues = []

        # Check which videos are valid
        for vid in tqdm(existing_videos, desc="Checking videos"):
            has_features = f"{vid}/features" in h5check
            has_gtscore = f"{vid}/gtscore" in h5check

            # Skip videos without features or gtscore
            if not has_features or not has_gtscore:
                videos_to_remove.append(vid)
                continue

            # Check if dimensions match
            try:
                features_length = h5check[f"{vid}/features"].shape[0]
                gtscore_length = h5check[f"{vid}/gtscore"].shape[0]

                if features_length != gtscore_length:
                    dimension_issues.append((vid, features_length, gtscore_length))
                    videos_to_remove.append(vid)
                else:
                    videos_to_keep.append(vid)
            except Exception as e:
                print(f"Error checking dimensions for {vid}: {str(e)}")
                videos_to_remove.append(vid)

    print(f"Found {len(existing_videos)} videos in total")
    print(f"Videos to keep: {len(videos_to_keep)}")
    print(f"Videos to remove: {len(videos_to_remove)}")
    print(f"Videos with dimension mismatches: {len(dimension_issues)}")

    if dimension_issues:
        print("Sample dimension issues:")
        for i, (vid, feat_len, gt_len) in enumerate(dimension_issues[:5]):
            print(f"  {vid}: features length = {feat_len}, gtscore length = {gt_len}")
        if len(dimension_issues) > 5:
            print(f"  ... and {len(dimension_issues) - 5} more")

    # If there are videos to remove, create a new clean h5 file
    if videos_to_remove:
        print("Creating new h5 file without problematic videos...")

        # Create a temporary file
        temp_h5_path = "temp_mr_hisum.h5"

        with h5py.File(h5_file_path, 'r') as old_h5, h5py.File(temp_h5_path, 'w') as new_h5:
            # Copy only valid videos
            for vid in tqdm(videos_to_keep, desc="Copying valid videos"):
                # Copy all datasets for this video
                for key in old_h5[vid].keys():
                    # Create the group if it doesn't exist
                    if vid not in new_h5:
                        new_h5.create_group(vid)

                    # Copy the dataset
                    old_h5.copy(f"{vid}/{key}", new_h5[vid])

        # Replace the old file with the new one
        os.replace(temp_h5_path, h5_file_path)
        print(f"Successfully removed {len(videos_to_remove)} problematic videos")

        # Print the last 10 videos kept for verification
        if videos_to_keep:
            print("Last 10 videos in the cleaned file:")
            sorted_videos = sorted(videos_to_keep)
            for vid in sorted_videos[-10:]:
                print(f"  {vid}")
    else:
        print("No videos need to be removed. All videos have features and matching dimensions.")


if __name__ == "__main__":
    preprocess()