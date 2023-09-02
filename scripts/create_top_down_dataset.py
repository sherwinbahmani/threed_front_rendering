import argparse
import os
from tqdm import tqdm
import shutil

def main(in_dir, out_dir, in_dir_labels, file_name='rendered_scene_256.png', out_file_name='0.png'):
    scene_names = os.listdir(in_dir)
    scene_names_labels = os.listdir(in_dir_labels)
    for scene_name in tqdm(scene_names):
        in_dir_scene = os.path.join(in_dir, scene_name)
        if not os.path.isdir(in_dir_scene) or scene_name not in scene_names_labels:
            continue
        out_dir_scene = os.path.join(out_dir, scene_name)
        os.makedirs(out_dir_scene, exist_ok=True)
        in_file_path = os.path.join(in_dir_scene, file_name)
        out_file_path = os.path.join(out_dir_scene, out_file_name)
        shutil.copy(in_file_path, out_file_path)       
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--in-dir",
            default="/path/to/data/3dfront/processed/bedrooms_without_lamps/",
        )
    parser.add_argument(
            "--in-dir-labels",
            default="/path/to/data/3dfront/processed/bedrooms_without_lamps_top/labels",
        )
    parser.add_argument(
            "--out-dir",
            default="/path/to/data/3dfront/processed/bedrooms_without_lamps_top/images",
        )
    args = parser.parse_args()
    main(args.in_dir, args.out_dir, args.in_dir_labels)