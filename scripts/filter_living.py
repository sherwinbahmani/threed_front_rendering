import argparse
import os
from tqdm import tqdm
import shutil

def main(in_dir, out_dir):
    path_txt="/path/to/code/ATISS/config/living.txt"
    with open(path_txt) as f:
        lines = f.readlines()
    filter_scenes = [line.rstrip('\n') for line in lines]
    scene_names = os.listdir(os.path.join(in_dir, "images"))
    for scene_name in tqdm(scene_names):
        if scene_name in filter_scenes or len(os.listdir(os.path.join(in_dir, 'images', scene_name))) != 40:
            continue
        for out_type in ['images', 'labels']:
            out_dir_images_scene = os.path.join(out_dir, out_type, scene_name)
            in_dir_images_scene = os.path.join(in_dir, out_type, scene_name)
            os.makedirs(out_dir_images_scene, exist_ok=True)
            shutil.copytree(in_dir_images_scene, out_dir_images_scene, dirs_exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="/path/to/data/3dfront/processed/living_room_2_fused",
    )
    parser.add_argument(
        "--out-dir",
        default="/path/to/data/3dfront/processed/living_room_2_fused_filter"
    )
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)
