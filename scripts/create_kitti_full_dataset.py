import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import PIL
import shutil

# Layout: (H,W): (forward, left/right)
# Uneven: grid_size=(256, 256), crop_size=(256, 128)

def main(in_dir, out_dir, in_dir_filter, label_name='labels', image_name='images', file_name='boxes.npz', grid_size=(256, 256), crop_size=(256, 256)):
    in_dir_labels = os.path.join(in_dir, label_name)
    out_dir_labels = os.path.join(out_dir, label_name)
    in_dir_images = os.path.join(in_dir, image_name)
    out_dir_images = os.path.join(out_dir, image_name)
    scene_names = sorted(os.listdir(in_dir_labels))
    # scene_names = [scene_name for scene_name in scene_names if "2013_05_28_drive_0000_sync" in scene_name] # Only 0 seq
    # scene_names = [scene_name for scene_name in scene_names if "2013_05_28_drive_0003_sync" not in scene_name and "2013_05_28_drive_0007_sync" not in scene_name] # All except 3 and 7
    if crop_size[0] != crop_size[1]:
        center_crop = torchvision.transforms.CenterCrop(crop_size)
    else:
        center_crop = None
    filters = np.load(in_dir_filter, allow_pickle=True)
    filter = filters["std_filter"].item()
    # filter = filters["strict_filter"].item()
    for scene_name in tqdm(scene_names):
        if not filter[scene_name]:
            continue
        in_dir_labels_scene = os.path.join(in_dir_labels, scene_name)
        out_dir_labels_scene = os.path.join(out_dir_labels, scene_name)
        in_dir_images_scene = os.path.join(in_dir_images, scene_name)
        out_dir_images_scene = os.path.join(out_dir_images, scene_name)
        in_file_path = os.path.join(in_dir_labels_scene, file_name)
        out_file_path = os.path.join(out_dir_labels_scene, file_name)
        os.makedirs(out_dir_labels_scene, exist_ok=True)
        os.makedirs(out_dir_images_scene, exist_ok=True)
        with open(in_file_path, 'rb') as f:
            boxes = np.load(f)
            intrinsic = boxes["intrinsic"][:3, :3]
            camera_coords = boxes["camera_coords"][:,[0,2,1]]
            target_coords = boxes["target_coords"][:,[0,2,1]]
            layout = boxes["layout"]
            layout_noveg = boxes["layout_noveg"]

        layout = torch.from_numpy(layout)
        layout_noveg = torch.from_numpy(layout_noveg)
        intrinsic = torch.from_numpy(intrinsic)
        camera_coords = torch.from_numpy(camera_coords)
        target_coords = torch.from_numpy(target_coords)
        if center_crop is None:
            layout = torch.nn.functional.interpolate(layout.unsqueeze(0),size=grid_size, mode='nearest').squeeze(0)
            layout_noveg = torch.nn.functional.interpolate(layout_noveg.unsqueeze(0),size=grid_size, mode='nearest').squeeze(0)
        else:
            layout = torch.nn.functional.interpolate(layout.unsqueeze(0),size=crop_size, mode='nearest').squeeze(0)
            layout_noveg = torch.nn.functional.interpolate(layout_noveg.unsqueeze(0),size=crop_size, mode='nearest').squeeze(0)
            layout = center_crop(layout)
            layout_noveg = center_crop(layout_noveg)

        labels = {
            'layout': layout,
            'layout_noveg': layout_noveg,
            'camera_coords': camera_coords,
            'target_coords': target_coords,
            'intrinsic': intrinsic,
            }
        # PIL.Image.fromarray(np.uint8(np.array(layout.permute(1,2,0).repeat(1,1,3)/max(layout.max(),1))*255), "RGB").save("image.png")
        for k, v in labels.items():
            labels[k] = v.numpy()
        np.savez(out_file_path, **labels)

        shutil.copytree(in_dir_images_scene, out_dir_images_scene, dirs_exist_ok=True)

        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="/localhome/sba229/data/kitti/full/kitti360_v1_512",
    )
    parser.add_argument(
        "--in-dir-filter",
        default="/localhome/sba229/data/kitti/full/kitti360_v1_512/filters/dset_dict.npz",
    )
    parser.add_argument(
        "--out-dir",
        default="/localhome/sba229/data/kitti/full/kitti360_v1_dset_dict_256_std"
    )
    args = parser.parse_args()
    main(args.in_dir, args.out_dir, args.in_dir_filter)