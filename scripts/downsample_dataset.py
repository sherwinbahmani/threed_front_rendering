import argparse
import os
from PIL import Image
from tqdm import tqdm

def main(in_dir, out_dir, out_res = (64, 64)):
    scene_names = os.listdir(in_dir)
    for scene_name in tqdm(scene_names):
        in_dir_scene = os.path.join(in_dir, scene_name)
        out_dir_scene = os.path.join(out_dir, scene_name)
        os.makedirs(out_dir_scene, exist_ok=True)
        img_names = os.listdir(in_dir_scene)
        for img_name in img_names:
            in_dir_img = os.path.join(in_dir_scene, img_name)
            out_dir_img = os.path.join(out_dir_scene, img_name)
            out_img = Image.open(in_dir_img).resize(out_res, Image.BILINEAR)
            out_img.save(out_dir_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="/path/to/data/3dfront/processed/bedrooms_without_lamps_full_7/images",
    )
    parser.add_argument(
        "--out-dir",
        default="/path/to/data/3dfront/processed/bedrooms_without_lamps_full_7_64/images"
    )
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)