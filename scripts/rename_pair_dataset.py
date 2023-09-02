import argparse
import os
from tqdm import tqdm
import numpy as np
import shutil

def main(in_dir, out_dir):
    scene_names = os.listdir(in_dir)
    for scene_name in tqdm(scene_names):
        in_dir_scene = os.path.join(in_dir, scene_name)
        out_dir_scene = os.path.join(out_dir, scene_name)
        os.makedirs(out_dir_scene, exist_ok=True)
        img_names = sorted(os.listdir(in_dir_scene))
        num_pairs = len(img_names) // 2
        img_names_out = img_names[:num_pairs] + [img_names[i].replace('.png', '_pair.png') for i in range(num_pairs)]
        for img_name_in, img_name_out in zip(img_names, img_names_out):
            in_file_path = os.path.join(in_dir_scene, img_name_in)
            out_file_path = os.path.join(out_dir_scene, img_name_out)
            if in_file_path != out_file_path:
                if in_dir_scene == out_dir_scene:
                    shutil.move(in_file_path, out_file_path)
                else:
                    shutil.copy(in_file_path, out_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="/path/to/data/3dfront/processed/bedrooms_without_lamps_full_8_without_walls_3_pair/images",
    )
    parser.add_argument(
        "--out-dir",
        default="/path/to/data/3dfront/processed/bedrooms_without_lamps_full_7_labels_pair_norm/images"
    )
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)