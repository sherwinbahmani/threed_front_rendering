import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import PIL
import shutil

def main(in_dir, out_dir, label_name='labels', image_name='images', file_name='boxes.npz'):
    in_dir_labels = os.path.join(in_dir, label_name)
    out_dir_labels = os.path.join(out_dir, label_name)
    in_dir_images = os.path.join(in_dir, image_name)
    out_dir_images = os.path.join(out_dir, image_name)
    scene_names = sorted(os.listdir(in_dir_labels))
    for scene_name in tqdm(scene_names):
        # # if "2013_05_28_drive_0002_sync_00008978" not in scene_name:
        # if "2013_05_28_drive_0004_sync_00009414" not in scene_name:
            
        #     continue
        in_dir_labels_scene = os.path.join(in_dir_labels, scene_name)
        out_dir_labels_scene = os.path.join(out_dir_labels, scene_name)
        in_file_path = os.path.join(in_dir_labels_scene, file_name)
        out_file_path = os.path.join(out_dir_labels_scene, file_name)
        in_dir_images_scene = os.path.join(in_dir_images, scene_name)
        out_dir_images_scene = os.path.join(out_dir_images, scene_name)
        os.makedirs(out_dir_images_scene, exist_ok=True)
        os.makedirs(out_dir_labels_scene, exist_ok=True)
        with open(in_file_path, 'rb') as f:
            boxes = np.load(f)
            intrinsic = boxes["intrinsic"]
            camera_coords = boxes["camera_coords"]
            target_coords = boxes["target_coords"]
            layout = boxes["layout"]
            layout_noveg = boxes["layout_noveg"]

        layout = torch.from_numpy(layout)
        layout_noveg = torch.from_numpy(layout_noveg)
        intrinsic = torch.from_numpy(intrinsic)
        camera_coords = torch.from_numpy(camera_coords)
        target_coords = torch.from_numpy(target_coords)
        
        # print(camera_coords)
        # print(target_coords)
        camera_coords = torch.nn.functional.interpolate(camera_coords.transpose(1,0).unsqueeze(0), scale_factor=3, mode='linear', align_corners=True).squeeze(0).transpose(1,0)
        target_coords = torch.nn.functional.interpolate(target_coords.transpose(1,0).unsqueeze(0), scale_factor=3, mode='linear', align_corners=True).squeeze(0).transpose(1,0)
        
        # print(camera_coords)
        # print(camera_coords.shape)
        # assert False

        labels = {
            'layout': layout,
            'layout_noveg': layout_noveg,
            'camera_coords': camera_coords,
            'target_coords': target_coords,
            'intrinsic': intrinsic,
            }
        for k, v in labels.items():
            labels[k] = v.numpy()
        np.savez(out_file_path, **labels)
        shutil.copytree(in_dir_images_scene, out_dir_images_scene, dirs_exist_ok=True)      
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="/path/to/data/kitti/kitti360_v1_dset_dict_256_strict",
    )
    parser.add_argument(
        "--out-dir",
        default="/path/to/data/kitti/kitti360_v1_dset_dict_256_strict_long"
    )
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)