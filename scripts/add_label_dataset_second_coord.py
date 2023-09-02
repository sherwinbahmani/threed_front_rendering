import argparse
import os
from tqdm import tqdm
import numpy as np

def main(in_dir, out_dir, file_name='boxes.npz'):
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
            floor_plan_vertices_calc = boxes["floor_plan_vertices_calc"]
        camera_coords = np.concatenate((camera_coords[:, :3], camera_coords[:, 3:]), axis=0)
        target_coords = np.concatenate((target_coords, target_coords), axis=0)
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
            'floor_plan_vertices_calc': floor_plan_vertices_calc,
        }
        np.savez(out_file_path, **labels)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="/path/to/data/3dfront/processed/bedrooms_without_lamps_full_7_labels_pair/labels",
    )
    parser.add_argument(
        "--out-dir",
        default="/path/to/data/3dfront/processed/bedrooms_without_lamps_full_7_labels_pair_concat"
    )
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)