import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
import shutil

def main(in_dir, out_dir, label_name='labels', file_name='boxes.npz'):
    scene_names = os.listdir(os.path.join(in_dir, label_name))
    for scene_name in tqdm(scene_names):
        in_dir_labels_scene = os.path.join(in_dir, label_name, scene_name)
        out_dir_labels_scene = os.path.join(out_dir, label_name, scene_name)
        in_file_path = os.path.join(in_dir_labels_scene, file_name)
        out_file_path = os.path.join(out_dir_labels_scene, file_name)
        if os.path.isfile(out_file_path):
            continue
        with open(in_file_path, 'rb') as f:
            boxes = np.load(f)
            class_labels=boxes["class_labels"]
            translations=boxes["translations"]
            sizes=boxes["sizes"]
            angles=boxes["angles"]
            room_layout = boxes["room_layout"]
            camera_coords=boxes["camera_coords"]
            target_coords = boxes["target_coords"]
            camera_coords_traj=boxes["camera_coords_traj"]
            target_coords_traj = boxes["target_coords_traj"]
        class_labels = torch.from_numpy(class_labels)
        translations = torch.from_numpy(translations)
        sizes = torch.from_numpy(sizes)
        angles = torch.from_numpy(angles)
        room_layout = torch.from_numpy(room_layout)
        camera_coords = torch.from_numpy(camera_coords)
        target_coords = torch.from_numpy(target_coords)
        camera_coords_traj = torch.from_numpy(camera_coords_traj)
        target_coords_traj = torch.from_numpy(target_coords_traj)

        # camera_coords = camera_coords[:len(camera_coords)//2]
        # target_coords = target_coords[:len(target_coords)//2]
        labels = {
            'class_labels': class_labels,
            'translations': translations,
            'sizes': sizes,
            'angles': angles,
            'room_layout': room_layout,
            'camera_coords': camera_coords.float(),
            'target_coords': target_coords.float(),
            }
        labels['camera_coords_traj'] = camera_coords_traj.float()
        labels['target_coords_traj'] = target_coords_traj.float()
        os.makedirs(out_dir_labels_scene, exist_ok=True)
        for k, v in labels.items():
            labels[k] = v.numpy()
            np.savez(out_file_path, **labels)

        # print("camera_coords", camera_coords.shape)
        # print(camera_coords)
        # print("target_coords", target_coords.shape)
        # print(target_coords)
        # print("camera_coords_traj", camera_coords_traj.shape)
        # print(camera_coords_traj)
        # print("target_coords_traj", target_coords_traj.shape)
        # print(target_coords_traj)
        # assert False


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="/path/to/data/3dfront/processed/bedrooms_without_lamps_full_9_fused",
    )
    parser.add_argument(
        "--out-dir",
        default="/path/to/data/3dfront/processed/lsun_9_labels_fused"
    )
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)