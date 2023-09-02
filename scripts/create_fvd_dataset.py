import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
import shutil

def main(in_dir, out_dir, images_name='images', start_frame_idx=20):
    scene_names = os.listdir(os.path.join(in_dir, images_name))
    for scene_name in tqdm(scene_names):
        in_dir_images_scene = os.path.join(in_dir, images_name, scene_name)
        out_dir_images_scene = os.path.join(out_dir, images_name, scene_name)
        os.makedirs(out_dir_images_scene, exist_ok=True)
        frame_names = sorted(os.listdir(in_dir_images_scene))
        for f_name in frame_names:
            f_name_idx, f_app = f_name.split(".")
            f_idx = int(f_name_idx) - start_frame_idx
            if f_idx < 0:
                continue
            f_name_out = str(f_idx).zfill(len(f_name_idx)) + f".{f_app}"
            in_dir_f = os.path.join(in_dir_images_scene, f_name)
            out_dir_f = os.path.join(out_dir_images_scene, f_name_out)
            shutil.copy(in_dir_f, out_dir_f)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="/path/to/data/3dfront/processed/living_room_2_fused_filter",
    )
    parser.add_argument(
        "--out-dir",
        default="/path/to/data/3dfront/processed/living_room_2_fused_filter_fvd"
    )
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)