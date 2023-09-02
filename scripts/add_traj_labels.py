import argparse
import os
from tqdm import tqdm
import numpy as np
import torch

def main(in_dir, out_dir, in_dir_traj, file_name='boxes.npz'):
    scene_names = os.listdir(in_dir)
    for scene_name in tqdm(scene_names):
        in_dir_scene = os.path.join(in_dir, scene_name)
        in_dir_traj_scene = os.path.join(in_dir_traj, scene_name)
        out_dir_scene = os.path.join(out_dir, scene_name)
        in_file_path = os.path.join(in_dir_scene, file_name)
        in_traj_file_path = os.path.join(in_dir_traj_scene, file_name)
        out_file_path = os.path.join(out_dir_scene, file_name)
        if os.path.isfile(out_file_path):
            continue

        if os.path.isdir(in_dir_scene) and os.path.isdir(in_dir_traj_scene):
            traj = True
        else:
            traj = False
        os.makedirs(out_dir_scene, exist_ok=True)
        with open(in_file_path, 'rb') as f:
            boxes = np.load(f)
            class_labels=boxes["class_labels"]
            translations=boxes["translations"]
            sizes=boxes["sizes"]
            angles=boxes["angles"]
            room_layout = boxes["room_layout"]
            camera_coords=boxes["camera_coords"]
            target_coords = boxes["target_coords"]
        if traj:
            with open(in_traj_file_path, 'rb') as f_traj:
                boxes_traj = np.load(f_traj)
                camera_coords_traj = boxes_traj["camera_coords"]
                target_coords_traj = boxes_traj["target_coords"]
        class_labels = torch.from_numpy(class_labels)
        translations = torch.from_numpy(translations)
        sizes = torch.from_numpy(sizes)
        angles = torch.from_numpy(angles)
        room_layout = torch.from_numpy(room_layout)
        camera_coords = torch.from_numpy(camera_coords)
        target_coords = torch.from_numpy(target_coords)
        labels = {
            'class_labels': class_labels,
            'translations': translations,
            'sizes': sizes,
            'angles': angles,
            'room_layout': room_layout,
            'camera_coords': camera_coords.float(),
            'target_coords': target_coords.float(),
            }
        if traj:
            camera_coords_traj = torch.from_numpy(camera_coords_traj)
            target_coords_traj = torch.from_numpy(target_coords_traj)
            labels['camera_coords_traj'] = camera_coords_traj.float()
            labels['target_coords_traj'] = target_coords_traj.float()
        
        for k, v in labels.items():
            labels[k] = v.numpy()
        np.savez(out_file_path, **labels)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="/path/to/data/3dfront/processed/bedrooms_without_lamps_full_7_labels_norm/",
    )
    parser.add_argument(
        "--in-dir-traj",
        default="/path/to/data/3dfront/processed/bedrooms_without_lamps_full_8_traj_diag_2_floor_red_4/labels",
    )
    parser.add_argument(
        "--out-dir",
        default="/path/to/data/3dfront/processed/bedrooms_without_lamps_full_7_both_coords_norm"
    )
    args = parser.parse_args()
    main(args.in_dir, args.out_dir, args.in_dir_traj)