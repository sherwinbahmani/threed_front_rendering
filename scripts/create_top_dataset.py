import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
import shutil

def main(in_dir, in_dir_ref, out_dir, label_name='labels', images_name='images', file_name='boxes.npz'):
    scene_names = os.listdir(os.path.join(in_dir, images_name))
    scene_names_ref = os.listdir(os.path.join(in_dir_ref, images_name))
    for scene_name in tqdm(scene_names):
        if scene_name not in scene_names_ref:
            continue
        in_dir_images_scene = os.path.join(in_dir, images_name, scene_name)
        out_dir_labels_scene = os.path.join(out_dir, label_name, scene_name)
        out_dir_images_scene = os.path.join(out_dir, images_name, scene_name)
        in_dir_labels_scene_ref = os.path.join(in_dir_ref, label_name, scene_name)
        in_ref_file_path = os.path.join(in_dir_labels_scene_ref, file_name)
        out_file_path = os.path.join(out_dir_labels_scene, file_name)
        os.makedirs(out_dir_images_scene, exist_ok=True)
        os.makedirs(out_dir_labels_scene, exist_ok=True)

        shutil.copy(in_ref_file_path, out_file_path)
        shutil.copytree(in_dir_images_scene, out_dir_images_scene, dirs_exist_ok=True)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="/path/to/data/3dfront/processed/bedrooms_without_lamps_full_8_traj_diag_2_floor_red_4_top",
    )
    parser.add_argument(
        "--in-dir-ref",
        default="/path/to/data/3dfront/processed/bedrooms_without_lamps_full_9_fused",
    )
    parser.add_argument(
        "--out-dir",
        default="/path/to/data/3dfront/processed/bedrooms_without_lamps_full_9_fused_top"
    )
    args = parser.parse_args()
    main(args.in_dir, args.in_dir_ref, args.out_dir)