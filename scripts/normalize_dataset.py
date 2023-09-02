import argparse
import os
from tqdm import tqdm
import torch
import numpy as np

def normalize_camera_coords(camera_coords, target_coords, max_coords):
    max_coords = torch.tensor(max_coords, device=camera_coords.device)
    # camera_coords = camera_coords/max_coords
    # target_coords = target_coords/max_coords
    # In case for two pairs repeated along column axis
    num_cams = camera_coords.shape[1]//3
    num_tar = target_coords.shape[1]//3
    camera_coords = camera_coords/max_coords.repeat(num_cams)
    target_coords = target_coords/max_coords.repeat(num_tar)
    camera_coords[..., [i*3 + 1 for i in range(num_cams)]] -= 0.5
    target_coords[..., [i*3 + 1 for i in range(num_tar)]] -= 0.5
    camera_coords *= 2
    target_coords *= 2
    assert camera_coords.min() >= -1 and camera_coords.max() <= 1, camera_coords
    assert target_coords.min() >= -1 and target_coords.max() <= 1, target_coords
    return camera_coords, target_coords

def main(in_dir, out_dir, max_coords, file_name='boxes.npz'):
    scene_names = os.listdir(in_dir)
    for scene_name in tqdm(scene_names):
        in_dir_scene = os.path.join(in_dir, scene_name)
        out_dir_scene = os.path.join(out_dir, scene_name)
        os.makedirs(out_dir_scene, exist_ok=True)
        in_file_path = os.path.join(in_dir_scene, file_name)
        out_file_path = os.path.join(out_dir_scene, file_name)

        with open(in_file_path, 'rb') as f:
            boxes = np.load(f)
            class_labels=boxes["class_labels"]
            translations=boxes["translations"]
            sizes=boxes["sizes"]
            angles=boxes["angles"]
            room_layout = boxes["room_layout"]
            camera_coords=boxes["camera_coords"]
            target_coords = boxes["target_coords"]
        class_labels = torch.from_numpy(class_labels)
        translations = torch.from_numpy(translations)
        sizes = torch.from_numpy(sizes)
        angles = torch.from_numpy(angles)
        room_layout = torch.from_numpy(room_layout)
        camera_coords = torch.from_numpy(camera_coords)
        target_coords = torch.from_numpy(target_coords)
        # Scale camera and target coords into [-1, 1]
        camera_coords, target_coords = normalize_camera_coords(camera_coords, target_coords, max_coords)
        # Divide by max values to scale into [-0.5,0.5]
        max_coords = torch.tensor(max_coords)
        sizes = sizes/max_coords
        translations = translations/max_coords
        translations[..., 1] -= 0.5
        room_layout = (room_layout / 255).permute(2, 0, 1)
        translations = translations.flip(1)
        sizes = sizes.flip(1)
        translations[..., 1] *= -1
        labels = {
            'class_labels': class_labels,
            'translations': translations,
            'sizes': sizes,
            'angles': angles,
            'room_layout': room_layout,
            'camera_coords': camera_coords.float(),
            'target_coords': target_coords.float()
        }
        for k, v in labels.items():
            labels[k] = v.numpy()
        np.savez(out_file_path, **labels)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="/path/to/data/3dfront/processed/living_room_2_fused_filter_traj_long/labels",
    )
    parser.add_argument(
        "--out-dir",
        default="/path/to/data/3dfront/processed/living_room_2_fused_filter_traj_long_norm/labels"
    )
    parser.add_argument(
        "--max-coords",
        # default=[6.0, 4.0, 6.0] # Bedroom
        default=[12.0, 4.0, 12.0] # Living
    )
    args = parser.parse_args()
    main(args.in_dir, args.out_dir, args.max_coords)