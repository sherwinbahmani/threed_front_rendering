import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
import shutil

def main(in_dir, in_dir_traj, out_dir, label_name='labels', images_name='images', file_name='boxes.npz'):
    scene_names = os.listdir(os.path.join(in_dir, label_name))
    scene_names_traj = os.listdir(os.path.join(in_dir_traj, label_name))
    for scene_name in tqdm(scene_names):
        if scene_name not in scene_names_traj:
            continue
        in_dir_labels_scene = os.path.join(in_dir, label_name, scene_name)
        in_dir_images_scene = os.path.join(in_dir, images_name, scene_name)
        out_dir_labels_scene = os.path.join(out_dir, label_name, scene_name)
        out_dir_images_scene = os.path.join(out_dir, images_name, scene_name)
        in_dir_labels_scene_traj = os.path.join(in_dir_traj, label_name, scene_name)
        in_dir_images_scene_traj = os.path.join(in_dir_traj, images_name, scene_name)
        in_file_path = os.path.join(in_dir_labels_scene, file_name)
        in_traj_file_path = os.path.join(in_dir_labels_scene_traj, file_name)
        out_file_path = os.path.join(out_dir_labels_scene, file_name)
        if not os.path.isdir(in_dir_images_scene) or not os.path.isdir(in_dir_images_scene_traj):
            continue
        if os.path.isfile(out_file_path) or len(os.listdir(in_dir_images_scene_traj)) != len(os.listdir(in_dir_images_scene)) and len(os.listdir(in_dir_images_scene)) != 0:
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
        with open(in_traj_file_path, 'rb') as f_traj:
            boxes_traj = np.load(f_traj)
            camera_coords_traj = boxes_traj["camera_coords"]
            target_coords_traj = boxes_traj["target_coords"]
        in_exists = os.path.isfile(in_file_path) and os.path.isfile(in_traj_file_path) and os.path.isdir(in_dir_images_scene) and os.path.isdir(in_dir_images_scene_traj)
        if all([np.equal(camera_coords_traj[0], coords).all() for coords in camera_coords_traj]) or not in_exists:
            print(f"Skip {scene_name}")
            continue
        class_labels = torch.from_numpy(class_labels)
        translations = torch.from_numpy(translations)
        sizes = torch.from_numpy(sizes)
        angles = torch.from_numpy(angles)
        room_layout = torch.from_numpy(room_layout)
        camera_coords = torch.from_numpy(camera_coords)
        target_coords = torch.from_numpy(target_coords)
        camera_coords_traj = torch.from_numpy(camera_coords_traj)
        target_coords_traj = torch.from_numpy(target_coords_traj)

        camera_coords = torch.cat((camera_coords, camera_coords_traj), 0)
        target_coords = torch.cat((target_coords, target_coords_traj), 0)
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

        image_file_names = sorted(os.listdir(in_dir_images_scene))
        n = len(image_file_names)
        os.makedirs(out_dir_images_scene, exist_ok=True)
        os.makedirs(out_dir_labels_scene, exist_ok=True)
        for f_name in image_file_names:
            f_name_idx, f_app = f_name.split(".")
            f_name_traj_out = str(int(f_name_idx)+n).zfill(len(f_name_idx)) + f".{f_app}"
            in_dir_traj_f = os.path.join(in_dir_images_scene_traj, f_name)
            out_dir_traj_f = os.path.join(out_dir_images_scene, f_name_traj_out)
            in_dir_f = os.path.join(in_dir_images_scene, f_name)
            out_dir_f = os.path.join(out_dir_images_scene, f_name)
            shutil.copy(in_dir_traj_f, out_dir_traj_f)
            shutil.copy(in_dir_f, out_dir_f)
            
        for k, v in labels.items():
            labels[k] = v.numpy()
        np.savez(out_file_path, **labels)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="/path/to/data/3dfront/processed/living_room_2_not_traj",
    )
    parser.add_argument(
        "--in-dir-traj",
        default="/path/to/data/3dfront/processed/living_room_2_traj",
    )
    parser.add_argument(
        "--out-dir",
        default="/path/to/data/3dfront/processed/living_room_2_fused"
    )
    args = parser.parse_args()
    main(args.in_dir, args.in_dir_traj, args.out_dir)