import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
import torchvision
from PIL import Image

def main(in_dir, out_dir, file_name='video.mp4'):
    scene_names = os.listdir(in_dir)
    for scene_name in tqdm(scene_names):
        in_dir_scene = os.path.join(in_dir, scene_name)
        out_dir_scene = os.path.join(out_dir, scene_name)
        os.makedirs(out_dir_scene, exist_ok=True)
        frame_names = sorted(os.listdir(in_dir_scene))
        # frame_names = sorted(frame_names, key=lambda fname: int(fname.split('_')[0])) # TODO: Only for blender
        frame_names = sorted(frame_names, key=lambda fname: int(fname.split('.')[0])) # TODO: Only for blender
        if len(frame_names) != 20:
            continue
        out_file_path = os.path.join(out_dir_scene, file_name)
        video = np.stack([np.array(Image.open(os.path.join(in_dir_scene, frame_name)).convert('RGB')) for frame_name in frame_names])
        torchvision.io.write_video(out_file_path, torch.from_numpy(video), fps=8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="/path/to/data/3dfront/processed/living_room_1/images",
    )
    parser.add_argument(
        "--out-dir",
        default="/path/to/data/3dfront/processed/living_room_1_videos"
    )
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)