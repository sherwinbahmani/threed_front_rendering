import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import scipy.ndimage
import scipy.spatial

CLASSES_3DFRONT_WITH_LAMPS = {
    0: 'armchair', 1: 'bookshelf', 2: 'cabinet', 3: 'ceiling_lamp', 4: 'chair', 5: 'children_cabinet', 6: 'coffee_table',
    7: 'desk', 8: 'double_bed', 9: 'dressing_chair', 10: 'dressing_table', 11: 'kids_bed', 12: 'night_stand',
    13: 'pendant_lamp', 14: 'shelf', 15: 'single_bed', 16: 'sofa', 17: 'stool', 18: 'table', 19: 'tv_stand', 20: 'wardrobe',
    21: 'start', 22: 'end'
}

CLASSES_3DFRONT_WITHOUT_LAMPS = {
    0: 'armchair', 1: 'bookshelf', 2: 'cabinet', 3: 'chair', 4: 'children_cabinet', 5: 'coffee_table',
    6: 'desk', 7: 'double_bed', 8: 'dressing_chair', 9: 'dressing_table', 10: 'kids_bed', 11: 'night_stand',
    12: 'shelf', 13: 'single_bed', 14: 'sofa', 15: 'stool', 16: 'table', 17: 'tv_stand', 18: 'wardrobe',
    19: 'start', 20: 'end'
}

CLASSES_3DFRONT = {23: CLASSES_3DFRONT_WITH_LAMPS, 21: CLASSES_3DFRONT_WITHOUT_LAMPS}

# import open3d as o3d
bed_class_labels = {
    21: [7, 10, 13], # without lamps
    23: [8, 11, 15], # with lamps
    }
def main(args):
    scenes = sorted(os.listdir(args.path_in))[args.start_idx:args.end_idx]
    for scene in tqdm(scenes):
        # if 'SecondBedroom-6482' not in scene:
        #     continue
        path_in_scene = os.path.join(args.path_in, scene)
        path_in_labels = os.path.join(path_in_scene, 'boxes.npz')
        path_out_scene = os.path.join(args.path_out, scene)
        path_out_labels = os.path.join(path_out_scene, 'boxes.npz')
        if not os.path.isfile(path_in_labels) or os.path.isfile(path_out_labels):
            continue
        labels = load_labels(path_in_labels, args.max_coords, beds_only=args.beds_only)
        if args.remove_classes is not None:
            num_classes = labels['class_labels'].shape[-1]
            class_dict = dict((v,k) for k,v in CLASSES_3DFRONT[num_classes].items())
            remove_classes_idx = [class_dict[class_name] for class_name in args.remove_classes]
            class_indices = np.where(labels['class_labels'].cpu().numpy())[1]
            remove_classes_mask = np.isin(class_indices, remove_classes_idx) == False
            remove_classes_mask = remove_classes_mask.tolist()
            labels['class_labels'] = labels['class_labels'][remove_classes_mask]
            labels['translations'] = labels['translations'][remove_classes_mask]
            labels['sizes'] = labels['sizes'][remove_classes_mask]
            labels['angles'] = labels['angles'][remove_classes_mask]

        scene_grid, bed_grid, room_layout, bed_center, valid_mask = create_voxel_grid(**labels, beds_only=args.beds_only, bed_size=args.bed_size)
        camera_coords, target_coords = sample_camera_positions(scene_grid, bed_grid, room_layout, bed_center, args.max_coords, args.num_samples_scene, beds_only=args.beds_only)
        if camera_coords is None or target_coords is None:
            print(f"{scene} does not have the bed or floorplan in the center")
            continue

        if target_coords.shape[0] == args.num_samples_scene and camera_coords.shape[0] == args.num_samples_scene:
            # Write camera and target coordinates into labels
            labels_out = dict(np.load(path_in_labels))
            labels_out['camera_coords'] = camera_coords.cpu().numpy()
            labels_out['target_coords'] = target_coords.cpu().numpy()
            if args.remove_classes is not None:
                # Remove objects not in class list
                for k in ['class_labels', 'translations', 'sizes', 'angles']:
                    labels_out[k] = labels_out[k][remove_classes_mask]
            # Remove objects not on the ground
            for k in ['class_labels', 'translations', 'sizes', 'angles']:
                labels_out[k] = labels_out[k][valid_mask]
            num_invalid = valid_mask.count(False)
            if num_invalid > 0:
                print(f'Found {num_invalid} objects not on the ground for {scene}')
            # Don't save if there is no bed
            num_classes = labels_out['class_labels'].shape[-1]
            classes = np.where(labels_out['class_labels'])[1]
            if np.isin(classes, bed_class_labels[num_classes]).sum() == 0:
                print(f'Found no bed for {scene}')
                continue
            os.makedirs(path_out_scene, exist_ok=True)
            np.savez(path_out_labels, **labels_out)
        

def sample_camera_positions(scene_grid, bed_grid, room_layout, bed_center, max_coords, num_samples_scene=10,
                            camera_height=1.7, seed=0, top_perc=0.5, dist_perc=0.10, beds_only=False):
    valid_coords_selected = torch.tensor([0., max_coords[1], 0.], device=scene_grid.device).unsqueeze(0).repeat(num_samples_scene, 1)
    target_coords_selected = torch.tensor([0., 0., 0.], device=scene_grid.device).unsqueeze(0).repeat(num_samples_scene, 1)
    if valid_coords_selected.shape[0] != num_samples_scene:
        print(f"Only found {valid_coords_selected.shape[0]} valid camera coordinates")
    if target_coords_selected.shape[0] != num_samples_scene:
        print(f"Only found {valid_coords_selected.shape[0]} valid target coordinates")
    return valid_coords_selected, target_coords_selected

# def sample_camera_positions(scene_grid, bed_grid, room_layout, max_coords, num_samples_scene=10,
#                             camera_height=1.7, num_samples_target=100, num_samples_camera=10000, seed=0,
#                             top_perc=0.001):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     # Remove batch dimension
#     scene_grid = scene_grid.squeeze(0)
#     bed_grid = bed_grid.squeeze(0)
#     room_layout = room_layout.squeeze(0)
#     H, D, W = scene_grid.shape
#     # Revert grid back to original shapes: mirror z axis (W)
#     scene_grid = torch.flip(scene_grid, [2])
#     bed_grid = torch.flip(bed_grid, [2])
#     # # Look at object on the floor
#     scene_layout = scene_grid[:,-1,:]
#     # Look at object taking up the whole y axis
#     # scene_layout = scene_grid.sum(dim=1) > 0
#     # TODO: Look at objects cutting off at specific height (camera position)
#     # Look for positions which don't have objects and are within the room layout
#     valid_layout = torch.logical_and(room_layout, scene_layout == False)
#     valid_indices = torch.stack(torch.where(valid_layout), dim=1)
#     # Convert indices back to coordinates
#     valid_coords = valid_indices / (H - 1)
#     # Shift x and z coordinate to be centered at 0 [-H/2,+H/2], [-W/2,+W/2]
#     valid_coords[:,0] -= 0.5
#     valid_coords[:,1] -= 0.5
#     # Multiply H and W scaling back
#     valid_coords[:, 0] *= max_coords[0]
#     valid_coords[:, 1] *= max_coords[2]
#     # Add camera height
#     camera_height_coords = torch.ones((valid_coords.shape[0], 1), device=valid_coords.device) * camera_height
#     valid_coords = torch.cat((valid_coords[:,[0]], camera_height_coords, valid_coords[:,[1]]), dim=1)
#     # Flip grid to revert back indexing from bottom floorplan to normal coordinates
#     bed_grid_flipped = torch.flip(bed_grid, [1])
#     # Find points inside bed bounding box
#     target_indices = torch.stack(torch.where(bed_grid_flipped), dim=1)
#     target_coords = target_indices / (H - 1)
#     # Shift x and z coordinate to be centered at 0 [-H/2,+H/2], [-W/2,+W/2]
#     target_coords[:,0] -= 0.5
#     target_coords[:,2] -= 0.5

#     # Scale coordinates back to absolute values
#     target_coords[:, 0] *= max_coords[0]
#     target_coords[:, 1] *= max_coords[1]
#     target_coords[:, 2] *= max_coords[2]

#     # torch.cartesian_prod(torch.arange(target_coords.shape[0]), torch.arange(valid_coords.shape[0]))
#     # Sample max number of coordinates from 
#     target_coords_sampled = target_coords[torch.randperm(target_coords.shape[0])[:num_samples_target]]
#     valid_coords_sampled = valid_coords[torch.randperm(valid_coords.shape[0])[:num_samples_camera]]
#     # Combine every target point with every camera point
#     indices_combined = torch.cartesian_prod(torch.arange(target_coords_sampled.shape[0]), torch.arange(valid_coords_sampled.shape[0]))
#     target_indices = indices_combined[:,0]
#     valid_indices = indices_combined[:,1]
#     l2_dist = (target_coords_sampled[target_indices] - valid_coords_sampled[valid_indices]).square().sum(dim=1).sqrt()
#     # Get largest l2 distances
#     l2_dist_indices = l2_dist.sort(descending=True)[1]
#     # Take the top x %
#     l2_dist_indices_top = l2_dist_indices[:int(l2_dist_indices.shape[0]*top_perc)]
#     # Randomly pick one index
#     index_selected = l2_dist_indices_top[torch.randperm(l2_dist_indices_top.shape[0])][:num_samples_scene]
#     target_coords_selected = target_coords_sampled[target_indices[index_selected]]
#     valid_coords_selected = valid_coords_sampled[valid_indices[index_selected]]
#     return valid_coords_selected, target_coords_selected

    # save_point_cloud(target_coords, "/path/to/code/ATISS/points_target.ply")
    # save_point_cloud(valid_coords, "/path/to/code/ATISS/points_camera.ply")


# def save_point_cloud(xyz, out_path):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz.cpu().numpy())
#     if os.path.isfile(out_path):
#         os.remove(out_path)
#     o3d.io.write_point_cloud(out_path, pcd)

def load_labels(path_file, max_coords, device='cuda', beds_only=False):
    with open(path_file, 'rb') as f:
        boxes = np.load(f)
        class_labels=boxes["class_labels"]
        translations=boxes["translations"]
        sizes=boxes["sizes"]
        angles=boxes["angles"]
        room_layout = boxes["room_layout"]
        floor_plan_centroid = boxes["floor_plan_centroid"]
    class_labels = torch.from_numpy(class_labels).to(device)
    translations = torch.from_numpy(translations).to(device)
    sizes = torch.from_numpy(sizes).to(device)
    angles = torch.from_numpy(angles).to(device)
    room_layout = torch.from_numpy(room_layout).to(device)
    floor_plan_centroid = torch.from_numpy(floor_plan_centroid).to(device)
    # Divide by max values to scale into [-0.5,0.5]
    sizes = sizes/torch.tensor(max_coords, device=sizes.device)
    translations = translations/torch.tensor(max_coords, device=sizes.device)
    translations[..., 1] -= 0.5
    room_layout = (room_layout / 255).permute(2, 0, 1)
    # Swap axes H and W
    room_layout = room_layout.permute(0, 2, 1)
    # Mirror z axis (W)
    translations[:,2] *= -1
    room_layout = room_layout.flip(dims=[2])
    if beds_only:
        # Set center to 0,0
        translations[:,[0, 2]] = 0
    labels = {
        'class_labels': class_labels,
        'translations': translations,
        'sizes': sizes,
        'angles': angles,
        'room_layout': room_layout,
        'floor_plan_centroid': floor_plan_centroid,
    }
    return labels

def create_voxel_grid(class_labels, translations, sizes, angles, room_layout, floor_plan_centroid, beds_only=False, bed_size=1.0, dims=(256, 256, 256), num_vertices = 256):
    """
    Sizes and translations need to be in [-0.5, 0.5]
    Args:
        class_labels: (OBJ, NUM_C)
        translations: (OBJ, 3=(x,y,z))
        sizes: (OBJ, 3=(x,y,z))
        angles: (OBJ, 1=(theta_z))
        room_layout: (1, H, W)
    Returns:
        scene_grid: (1, H, D, W)
        bed_grid: (1, H, D, W)
        room_layout: (1, H, W)
        bed_center: (3=(x, y, z))
    """
    voxel_mask = torch.zeros((1, *dims), device=sizes.device, dtype=sizes.dtype)
    voxel_masks = torch.zeros((1, *dims), device=sizes.device, dtype=sizes.dtype)
    # Sort by largest objects
    obj_indices = torch.prod(sizes, 1).sort(descending=True)[1]
    # Find bed index
    bed_indices_mask = class_labels[:,bed_class_labels[class_labels.shape[-1]]]
    bed_indices_mask_objs = bed_indices_mask.sum(dim=1)
    # assert bed_indices_mask_objs.sum() == 1, f"Found {int(bed_indices_mask_objs.max())} beds in the scene"
    # bed_index = bed_indices_mask_objs.argmax()
    # Pick largest bed if there are multiple
    bed_index = obj_indices[(bed_indices_mask_objs==1)[obj_indices]][0]
    bed_grid = None
    bed_center = None
    valid_mask = []
    for j in obj_indices:
        if beds_only and j != bed_index:
            continue
        size = sizes[j]
        if j == bed_index and bed_size != 1.0:
            size = size * bed_size
        if torch.count_nonzero(size) == 0:
            break
        translation = translations[j]
        R = get_rotation_matrix(angles[j])
        xp = torch.linspace(- size[0], size[0], num_vertices, device=sizes.device)
        yp = torch.linspace(- size[1], size[1], num_vertices, device=sizes.device)
        zp = torch.linspace(- size[2], size[2], num_vertices, device=sizes.device)
        coords = torch.stack(torch.meshgrid(xp,yp,zp, indexing='ij')).view(3, -1)
        coords = torch.mm(R[0].T, coords) + translation.unsqueeze(-1)
        # Start y axis from bottom of the voxel grid to set floorplan at the bottom
        coords[1] *= -1
        # Clamp because of numerical precision
        coords = coords.clamp(-0.5, 0.5)
        occ_grid = voxelize(coords.transpose(1,0).unsqueeze(0), dims[0]).long()
        voxel_mask = torch.logical_and(voxel_masks == 0, occ_grid != 0)
        voxel_masks += voxel_mask

        # Valid object if it is on the floor
        if voxel_mask[:, :, -1, :].sum() > 0:
            valid_mask.append(True)
        else:
            valid_mask.append(False)

        # Check if object is bed
        if j == bed_index:
            bed_grid = voxel_mask
            bed_center = translation.clone()

    room_layout = torch.nn.functional.interpolate(room_layout.unsqueeze(0),
                                                            size=(dims[0], dims[2]),
                                                            mode='nearest').squeeze(0)
    return voxel_masks, bed_grid, room_layout, bed_center, valid_mask

def get_rotation_matrix(theta):
    R = torch.zeros((1, 3, 3), device=theta.device)
    R[:, 0, 0] = torch.cos(theta)
    R[:, 0, 2] = torch.sin(theta)
    R[:, 2, 0] = -torch.sin(theta)
    R[:, 2, 2] = torch.cos(theta)
    R[:, 1, 1] = 1.
    return R

def voxelize(pc: torch.Tensor, voxel_size: int, grid_size=1., filter_outlier=True):
    b, n, _ = pc.shape
    half_size = grid_size / 2.
    valid = (pc >= -half_size) & (pc <= half_size)
    valid = torch.all(valid, 2)
    pc_grid = (pc + half_size) * (voxel_size - 1.) / grid_size  # thanks @heathentw
    indices_floor = torch.floor(pc_grid)
    indices = indices_floor.long()
    batch_indices = torch.arange(b, device=pc.device)
    batch_indices = shape_padright(batch_indices)
    batch_indices = torch.tile(batch_indices, (1, n)) #nnt.utils.tile(batch_indices, (1, n))
    batch_indices = shape_padright(batch_indices)
    indices = torch.cat((batch_indices, indices), 2)
    indices = torch.reshape(indices, (-1, 4))
    
    r = pc_grid - indices_floor
    rr = (1. - r, r)
    if filter_outlier:
        valid = valid.flatten()
        indices = indices[valid]
    
    if valid.sum() == 0:
        return torch.zeros((b, grid_size, grid_size, grid_size), device=pc.device, dtype=torch.bool)

    def interpolate_scatter3d(pos):
        updates_raw = rr[pos[0]][..., 0] * rr[pos[1]][..., 1] * rr[pos[2]][..., 2]
        updates = updates_raw.flatten()

        if filter_outlier:
            updates = updates[valid]

        indices_shift = torch.tensor([[0] + pos], device=pc.device)
        indices_loc = indices + indices_shift
        out_shape = (b,) + (voxel_size,) * 3
        out = torch.zeros(*out_shape, device=pc.device).flatten()
        rav_ind = ravel_index(indices_loc.t(), out_shape, pc.device).long()
        rav_ind = rav_ind.clamp(0, voxel_size**3 - 1) #avoid interpolation with indices outside of grid
        voxels = out.scatter_add_(-1, rav_ind, updates).view(*out_shape)
        return voxels

    voxels = [interpolate_scatter3d([k, j, i]) for k in range(2) for j in range(2) for i in range(2)]
    voxels = sum(voxels)
    voxels = torch.clamp(voxels, 0., 1.)
    voxels = voxels > 0.5
    return voxels

# Source: neuralnet_pytorch
def ravel_index(indices, shape, device):
    assert len(indices) == len(shape), 'Indices and shape must have the same length'
    shape = torch.tensor(shape, device=device, dtype=torch.long)
    return sum([indices[i] * torch.prod(shape[i + 1:]) for i in range(len(shape))])

def shape_padright(x, n_ones=1):
    pattern = tuple(range(x.ndimension())) + ('x',) * n_ones
    return dimshuffle(x, pattern)

def dimshuffle(x, pattern):
    assert isinstance(pattern, (list, tuple)), 'pattern must be a list/tuple'
    no_expand_pattern = [x for x in pattern if x != 'x']
    y = x.permute(*no_expand_pattern)
    shape = list(y.shape)
    for idx, e in enumerate(pattern):
        if e == 'x':
            shape.insert(idx, 1)
    return y.view(*shape)

def bwdist_manhattan(a, seedval=1):
    seed_mask = a==seedval
    z = np.argwhere(seed_mask)
    nz = np.argwhere(~seed_mask)

    out = np.zeros(a.shape, dtype=int)
    out[tuple(nz.T)] = scipy.spatial.distance.cdist(z, nz, 'cityblock').min(0).astype(int)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--path-in",
            default="/path/to/data/3dfront/processed/bedrooms_without_lamps/",
            help="Path to scenes with bounding boxes"
        )
    parser.add_argument(
            "--path-out",
            default="/path/to/data/3dfront/processed/bedrooms_without_lamps_top/labels",
            help="Path to scenes with bounding boxes"
        )
    parser.add_argument(
            "--max-coords",
            default=[6.0, 4.0, 6.0],
            help="Maximum xyz coordinate values"
        )
    parser.add_argument(
            "--num-samples-scene",
            default=1,
            type=int,
            help="Sample given number of different camera poses"
        )
    parser.add_argument(
        "--beds-only",
        action="store_true",
        help="Only render beds in the center"
    )
    parser.add_argument(
            "--start-idx",
            default=0,
            type=int
        )
    parser.add_argument(
            "--end-idx",
            default=10000,
            type=int
        )
    parser.add_argument(
            "--remove-classes",
            default=None,
            type=str,
            nargs='+'
        )
    parser.add_argument(
            "--bed-size",
            default=1.0,
            type=float
        )
    args = parser.parse_args()
    main(args)