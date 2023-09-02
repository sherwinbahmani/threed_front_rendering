import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
import shutil
import PIL.Image

def main(in_dir, out_dir, label_name='labels', images_name='images'):
    scene_names = os.listdir(os.path.join(in_dir, images_name))
    # scene_names = ["0047c3ab-951b-4182-9082-b9fbf099c142_SecondBedroom-201"]
    for scene_name in tqdm(scene_names):
        in_dir_images_scene = os.path.join(in_dir, images_name, scene_name)
        img_names = os.listdir(in_dir_images_scene)
        is_black = False
        for img_name in img_names:
            img_path = os.path.join(in_dir_images_scene, img_name)
            img = np.array(PIL.Image.open(img_path).convert('RGB'))
            # Black when at least 30% of pixels are black
            if (img==0).all(-1).sum() > img.shape[0]*img.shape[1]*0.3:
                is_black = True
                break
        if is_black:
            continue
        out_dir_labels_scene = os.path.join(out_dir, label_name, scene_name)
        out_dir_images_scene = os.path.join(out_dir, images_name, scene_name)
        in_dir_labels_scene = os.path.join(in_dir, label_name, scene_name)
        os.makedirs(out_dir_images_scene, exist_ok=True)
        os.makedirs(out_dir_labels_scene, exist_ok=True)

        shutil.copytree(in_dir_images_scene, out_dir_images_scene, dirs_exist_ok=True)
        shutil.copytree(in_dir_labels_scene, out_dir_labels_scene, dirs_exist_ok=True)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="/path/to/data/3dfront/processed/bedrooms_without_lamps_full_11_fused",
    )
    parser.add_argument(
        "--out-dir",
        default="/path/to/data/3dfront/processed/bedrooms_without_lamps_full_11_fused_filter"
    )
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)