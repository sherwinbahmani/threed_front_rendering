import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
import cv2 as cv

def main(in_dir, out_dir, file_name='boxes.npz', offset_m = 0.02352941, std = 0.001): # offset_m calculated for 6x6 room absolute value
    scene_names = os.listdir(in_dir)
    for scene_name in tqdm(scene_names):
        in_dir_scene = os.path.join(in_dir, scene_name)
        out_dir_scene = os.path.join(out_dir, scene_name)
        os.makedirs(out_dir_scene, exist_ok=True)
        in_file_path = os.path.join(in_dir_scene, file_name)
        out_file_path = os.path.join(out_dir_scene, file_name)

        with open(in_file_path, 'rb') as f:
            boxes = np.load(f)
            class_labels=boxes["class_labels"]
            translations=boxes["translations"]
            sizes=boxes["sizes"]
            angles=boxes["angles"]
            room_layout = boxes["room_layout"]
            camera_coords=boxes["camera_coords"]
            target_coords = boxes["target_coords"]
            floor_plan_centroid = boxes["floor_plan_centroid"]
            floor_plan_vertices = boxes["floor_plan_vertices"]

        camera_coords_second = np.copy(camera_coords)
        camera_offset_sign = (2 * np.random.randint(0, 2, size=(camera_coords.shape[0], 2)) - 1)
        camera_coords_second[:, [0, 2]] += (offset_m + np.random.randn(camera_coords.shape[0], 2) * std) * camera_offset_sign
        camera_coords = np.concatenate((camera_coords, camera_coords_second), axis=1)
        labels = {
            'class_labels': class_labels,
            'translations': translations,
            'sizes': sizes,
            'angles': angles,
            'room_layout': room_layout,
            'camera_coords': camera_coords,
            'target_coords': target_coords,
            'floor_plan_centroid': floor_plan_centroid,
            'floor_plan_vertices': floor_plan_vertices,
        }
        np.savez(out_file_path, **labels)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="/path/to/data/3dfront/processed/bedrooms_without_lamps_full_7_labels",
    )
    parser.add_argument(
        "--out-dir",
        default="/path/to/data/3dfront/processed/bedrooms_without_lamps_full_7_labels_pair/labels"
    )
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)