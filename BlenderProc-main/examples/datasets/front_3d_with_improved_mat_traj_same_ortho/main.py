import blenderproc as bproc
import argparse
import os
import numpy as np
import random
import re

def point_inside_polygon(point, poly, include_edges=True):
    '''
    Test if point (x,y) is inside polygon poly.

    poly is N-vertices polygon defined as 
    [(x1,y1),...,(xN,yN)] or [(x1,y1),...,(xN,yN),(x1,y1)]
    (function works fine in both cases)

    Geometrical idea: point is inside polygon if horisontal beam
    to the right from point crosses polygon even number of times. 
    Works fine for non-convex polygons.
    '''
    x, y = point
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if p1y == p2y:
            if y == p1y:
                if min(p1x, p2x) <= x <= max(p1x, p2x):
                    # point is on horisontal edge
                    inside = include_edges
                    break
                elif x < min(p1x, p2x):  # point is to the left from current edge
                    inside = not inside
        else:  # p1y!= p2y
            if min(p1y, p2y) <= y <= max(p1y, p2y):
                xinters = (y - p1y) * (p2x - p1x) / float(p2y - p1y) + p1x

                if x == xinters:  # point is right on the edge
                    inside = include_edges
                    break

                if x < xinters:  # point is to the left from current edge
                    inside = not inside

        p1x, p1y = p2x, p2y

    return inside

def fov_to_intrinsics(fov_degrees):
    focal_length = (1 / (np.tan(fov_degrees * 3.14159 / 360) * 2))
    intrinsics = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    intrinsics[0,0] = focal_length
    intrinsics[1,1] = focal_length
    return intrinsics

def read_camera_labels(path_labels, file='boxes.npz'):
    path_labels_file = os.path.join(path_labels, file)
    with open(path_labels_file, 'rb') as f:
        boxes = np.load(f)
        camera_coords = boxes["camera_coords"]
        target_coords = boxes["target_coords"]
        floor_plan_centroid = boxes["floor_plan_centroid"]
        floor_plan_vertices = boxes["floor_plan_vertices"]
        floor_plan_vertices_calc = boxes["floor_plan_vertices_calc"]
    return camera_coords, target_coords, floor_plan_centroid, floor_plan_vertices, floor_plan_vertices_calc
    
def main(front, future_folder, front_3D_texture_path, cc_material_path, path_labels, output_dir, render_background):
    if not os.path.exists(front) or not os.path.exists(future_folder):
        raise Exception("One of the two folders does not exist!")

    bproc.init()
    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    # set the light bounces
    bproc.renderer.set_light_bounces(diffuse_bounces=0, glossy_bounces=0, max_bounces=0,
                                    transmission_bounces=0, transparent_max_bounces=0)
    # load the front 3D objects
    loaded_objects = bproc.loader.load_front3d(
        json_path=front,
        future_model_path=future_folder,
        front_3D_texture_path=front_3D_texture_path,
        label_mapping=mapping
    )
    camera_coords_all, target_coords_all, floor_plan_centroid, floor_plan_vertices, floor_plan_vertices_calc = read_camera_labels(path_labels)
    floor_plan_centroid = floor_plan_centroid[[0, 2, 1]]
    floor_plan_vertices = floor_plan_vertices[:, [0, 2]]
    v_min = floor_plan_vertices.min(0)
    v_max = floor_plan_vertices.max(0)
    eps_all = 0.0
    eps_wall = 0.01
    for obj in loaded_objects:
        obj_box = obj.get_bound_box()[:, :-1]
        b_c = obj_box.mean(0)
        box_out_floor = b_c[0] < v_min[0] or b_c[0] > v_max[0] or b_c[1] < v_min[1] or b_c[1] > v_max[1]
        eps = eps_wall if 'Wall' in obj.get_name() else eps_all
        box_out_floor = box_out_floor and sum([np.allclose(b_c[0], v_min[0], atol=eps), np.allclose(b_c[0], v_max[0], atol=eps), np.allclose(b_c[1], v_min[1], atol=eps), np.allclose(b_c[1], v_max[1], atol=eps)]) == 0
        if render_background:
            # Full scene reduced
            rem_obj_list = ['Lamp', 'Ceiling', 'Cornice', 'Back', 'Front', 'Light', 'Slab']
        else:
            # White backgrounds new
            rem_obj_list = ['Lamp', 'Ceiling', 'Cornice', 'Wall', 'Pocket', 'Hole', 'Door', 'Baseboard', 'Window', 'wainscot', 'Back', 'Front', 'Light', 'Slab']
        rem_obj_list = ['Lamp', 'Ceiling', 'Cornice', 'Back', 'Front', 'Light', 'Slab', 'Window']
        if any(obj_name_rem in obj.get_name() for obj_name_rem in rem_obj_list):
            obj.hide()
        elif len(re.findall(r'[\u4e00-\u9fff]+', obj.get_name())) > 0:
            obj.hide()
        # Detect floors outside
        elif not point_inside_polygon(b_c, floor_plan_vertices_calc):
            obj.hide()
        elif box_out_floor:
            obj.hide()      
        
    cc_materials = bproc.loader.load_ccmaterials(cc_material_path, ["Bricks", "Wood", "Carpet", "Tile", "Marble"])

    # Same material to every wall
    floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
    floor_material = random.choice(cc_materials)
    for floor in floors:
        # For each material of the object
        for i in range(len(floor.get_materials())):
            floor.set_material(i, floor_material)

    fov = 70
    K = fov_to_intrinsics(fov)
    image_width = image_height = 256
    clip_start = 0.1
    clip_end = 1000
    K[0,0] *=image_width
    K[0,2] *=image_width
    K[1,1] *=image_height
    K[1,2] *=image_height
    bproc.python.camera.CameraUtility.set_intrinsics_from_K_matrix(K, image_width, image_height, clip_start, clip_end)

    bproc.renderer.set_output_format(enable_transparency=True)
    for camera_coords, target_coords in zip(camera_coords_all, target_coords_all):
        camera_coords = camera_coords[[0, 2, 1]]
        target_coords = target_coords[[0, 2, 1]]
        camera_coords = floor_plan_centroid + camera_coords
        target_coords = floor_plan_centroid + target_coords
        camera_coords = floor_plan_centroid
        target_coords = floor_plan_centroid
        camera_coords[-1] = 4.
        rotation_matrix = bproc.camera.rotation_from_forward_vec(target_coords - camera_coords)
        cam2world_matrix = bproc.math.build_transformation_mat(camera_coords, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)

    # Also render normals
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_segmentation_output(map_by=["category_id"])

    # render the whole pipeline
    data = bproc.renderer.render()

    # write the data to a .hdf5 container
    bproc.writer.write_hdf5(output_dir, data)

parser = argparse.ArgumentParser()
parser.add_argument("front", help="Path to the 3D front file")
parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
parser.add_argument("front_3D_texture_path", help="Path to the 3D FRONT texture folder.")
parser.add_argument('cc_material_path', nargs='?', default="resources/cctextures", help="Path to CCTextures folder, see the /scripts for the download script.")
parser.add_argument('path_labels', help="Path to camera labels")
parser.add_argument("output_dir", help="Path to where the data should be saved")
parser.add_argument("scene_idx", help="Scene index in camera labels", type=int)
parser.add_argument("--render_background", type=bool, default=False)
args = parser.parse_args()

scenes_front = sorted(os.listdir(args.front))
scene_labels = sorted(os.listdir(args.path_labels))
scene_label = scene_labels[args.scene_idx]
scene_front_label = [scene_front for scene_front in scenes_front if scene_front.split('.json')[0] in scene_label]
assert len(scene_front_label) == 1
scene_front_label = scene_front_label[0]

path_front = os.path.join(args.front, scene_front_label)
path_labels = os.path.join(args.path_labels, scene_label)
output_dir = os.path.join(args.output_dir, scene_label)
print(path_front)
if os.path.exists(output_dir):
    exit()
main(path_front, args.future_folder, args.front_3D_texture_path, args.cc_material_path, path_labels, output_dir, args.render_background)