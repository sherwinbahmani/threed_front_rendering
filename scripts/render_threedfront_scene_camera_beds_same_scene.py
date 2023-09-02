"""Script used for parsing 3D_FRONT scenes"""
import argparse
from genericpath import isfile
import logging
import os
import sys
from typing import List
from tqdm import tqdm
import math

import numpy as np
from PIL import Image
import pyrr
import trimesh
from scene_synthesis.datasets.threed_front import ThreedFront
from scene_synthesis.datasets import filter_function

from simple_3dviz import Scene, Mesh
from simple_3dviz.behaviours.keyboard import SnapshotOnKey
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.renderables.textured_mesh import TexturedMesh
from simple_3dviz.utils import render
from simple_3dviz.window import show

from utils import get_floor_plan, get_walls, export_scene, DirLock, \
    get_windows, get_doors, get_textured_trimesh


def ensure_parent_directory_exists(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def scene_init(mesh, up_vector, camera_position, camera_target, background):
    def inner(scene):
        scene.background = background
        scene.up_vector = up_vector
        scene.camera_position = camera_position
        scene.camera_target = camera_target
        scene.light = camera_position
        if mesh is not None:
            scene.add(mesh)
    return inner

def main(argv):
    parser = argparse.ArgumentParser(
        description="Visualize a 3D-FRONT room from json file"
    )
    parser.add_argument(
        "output_directory",
        help="Path to output directory"
    )
    parser.add_argument(
        "path_to_3d_front_dataset_directory",
        help="Path to the 3D-FRONT dataset"
    )
    parser.add_argument(
        "path_to_3d_future_dataset_directory",
        help="Path to the 3D-FUTURE dataset"
    )
    parser.add_argument(
        "path_to_model_info",
        help="Path to the 3D-FUTURE model_info.json file"
    )
    parser.add_argument(
        "path_to_floor_plan_textures",
        help="Path to floor texture images"
    )
    parser.add_argument(
        "path_to_walls_textures",
        help="Path to walls texture images"
    )
    parser.add_argument(
        "--path_to_door_textures",
        default="../door_textures/door_texture_images",
        help="Path to doors texture images"
    )
    parser.add_argument(
        "--path_to_labels",
        help="Path to boxes.npz per scene"
    )
    parser.add_argument(
        "--scene_id",
        default=None,
        help="The scene id of the scene to be visualized"
    )
    parser.add_argument(
        "--annotation_file",
        default="../config/bedroom_threed_front_splits.csv",
        help="Path to the train/test splits file"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="1,1,1,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,1,0",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="256,256",
        help="Define the size of the scene and the window"
    )
    parser.add_argument(
        "--save_frames",
        help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--with_screen",
        action="store_true",
        help="Show on screen"
    )
    parser.add_argument(
        "--with_orthographic_projection",
        action="store_true",
        help="Use orthographic projection"
    )
    parser.add_argument(
        "--with_floor_layout",
        action="store_true",
        help="Visualize also the rooom's floor"
    )
    parser.add_argument(
        "--with_walls",
        action="store_true",
        help="Visualize also the rooom's walls"
    )
    parser.add_argument(
        "--with_windows",
        action="store_true",
        help="Visualize also the rooom's windows"
    )
    parser.add_argument(
        "--with_doors",
        action="store_true",
        help="Visualize also the rooom's doors"
    )
    parser.add_argument(
        "--with_texture",
        action="store_true",
        help="Visualize objects with texture"
    )
    parser.add_argument(
        "--without_lamps",
        action="store_true",
        help="If set ignore lamps when rendering the room"
    )
    parser.add_argument(
        "--path_to_invalid_scene_ids",
        default="../config/invalid_threed_front_rooms.txt",
        help="Path to invalid scenes"
    )
    parser.add_argument(
        "--path_to_invalid_bbox_jids",
        default="../config/black_list.txt",
        help="Path to objects that ae blacklisted"
    )
    parser.add_argument(
        "--dataset_filtering",
        default="threed_front_bedroom",
        choices=[
            "threed_front_bedroom",
            "threed_front_livingroom",
            "threed_front_diningroom",
            "threed_front_library"
        ],
        help="The type of dataset filtering to be used"
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=70.,
    )
    parser.add_argument(
        "--total-idx",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--num-images",
        type=float,
        default=math.inf,
    )
    parser.add_argument(
        "--num-color-images",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0.001,6.0,0.001",
        help="Camer position in the scene"
    )
    args = parser.parse_args(argv)
    logging.getLogger("trimesh").setLevel(logging.ERROR)
    # Read labels
    scenes_labels = os.listdir(args.path_to_labels)
    scenes_labels_short = {path.split('_')[-1]: path for path in scenes_labels}

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size)
    if args.with_orthographic_projection:
        scene.camera_matrix = pyrr.Matrix44.orthogonal_projection(
            left=-3.1, right=3.1, bottom=-3.1, top=3.1, near=0.1, far=1000
        )
    else:
        scene.camera_matrix = pyrr.Matrix44.perspective_projection(
            args.fov, 1., 0.1, 1000.
            ).astype(np.float32)
    
    config = {
        "filter_fn":                 args.dataset_filtering,
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids": args.path_to_invalid_scene_ids,
        "path_to_invalid_bbox_jids": args.path_to_invalid_bbox_jids,
        "annotation_file":           args.annotation_file
    }

    d = ThreedFront.from_dataset_directory(
        args.path_to_3d_front_dataset_directory,
        args.path_to_model_info,
        args.path_to_3d_future_dataset_directory,
        path_to_room_masks_dir=None,
        path_to_bounds=None,
        # filter_fn=lambda s: s
        filter_fn=filter_function(config, ["train", "val", "test"], args.without_lamps)
    )
    # print("Loading dataset with {} rooms".format(len(d)))
    total_idx = 0
    for s in tqdm(d.scenes):
        behaviours = []
        if total_idx < args.total_idx or total_idx >= args.total_idx + args.num_images:
            total_idx += 1
            continue
        if (s.scene_id == args.scene_id or args.scene_id is None) and s.scene_id in scenes_labels_short:
            # Read camera and target position from label file
            label_scene_name = scenes_labels_short[s.scene_id]
            path_label_scene = os.path.join(args.path_to_labels, label_scene_name, 'boxes.npz')
            # if 'MasterBedroom-52264' not in s.scene_id:
            #     continue
            # if 'SecondBedroom-6482' not in s.scene_id:
            #     continue
            if not os.path.isfile(path_label_scene):
                print(f"File {path_label_scene} not found")
                continue
            path_to_scene = os.path.join(args.output_directory, s.uid)
            if os.path.exists(path_to_scene):
                print(f"Scene {path_to_scene} already exists")
                continue
            path_to_image = os.path.join(path_to_scene, "{:03d}.png")
            total_idx += 1
            # print("s.scene_id", s.scene_id)
            for color_index in range(args.num_color_images):
                for frame_index in range(args.frame_index):
                    img_index = (frame_index + 1) + args.frame_index * color_index
                    if os.path.isfile(path_to_image.format(img_index)):
                        continue
                    os.makedirs(path_to_scene, exist_ok=True)
                    # print(f"Rendering scene {s.scene_id} with frame index {frame_index}")
                    label_scene = np.load(path_label_scene)
                    args.camera_target = tuple(label_scene['target_coords'][frame_index])
                    args.camera_position = tuple(label_scene['camera_coords'][frame_index])
                    # print("args.camera_position", args.camera_position, "args.camera_target", args.camera_target)
                    scene.light = args.camera_position
                    path_to_file = os.path.join(
                        args.output_directory, s.scene_id
                    )
                    # Check optimistically if the file already exists
                    # if os.path.exists(path_to_file):
                    #     continue
                    # ensure_parent_directory_exists(path_to_file)
                    # Make sure we are the only ones creating this file
                    with DirLock(path_to_file + ".lock") as lock:
                        if not lock.is_acquired:
                            continue
                        # if os.path.exists(path_to_file):
                        #     continue
                        renderables = s.furniture_renderables(
                            with_floor_plan_offset=True, with_texture=args.with_texture
                        )
                        trimesh_meshes = []
                        trimesh_names = []
                        models_class_labels = []
                        renderables_filtered = []
                        for i, furniture in enumerate(s.bboxes):
                            if 'bed' not in furniture.label:
                                continue
                            renderables[i].translate(np.array([s.centroid[0]-furniture.position[0], 0, s.centroid[2]-furniture.position[2]]))
                            # furniture.z_angle = 0
                            furniture.position = s.centroid
                            renderables_filtered.append(renderables[i])
                            model_tag = furniture.raw_model_path.split("/")[-2]
                            models_class_labels.append(f"{furniture.label}:{model_tag}")
                            trimesh_names.append(f"object_{i:03d}")
                            
                            # Load the furniture and scale it as it is given in the dataset
                            raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
                            raw_mesh.scale(furniture.scale)

                            # Create a trimesh object for the same mesh in order to save
                            # everything as a single scene
                            tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
                            tr_mesh.visual.material.image = Image.open(
                                furniture.texture_image_path
                            )
                            tr_mesh.vertices *= furniture.scale
                            theta = furniture.z_angle
                            R = np.zeros((3, 3))
                            R[0, 0] = np.cos(theta)
                            R[0, 2] = -np.sin(theta)
                            R[2, 0] = np.sin(theta)
                            R[2, 2] = np.cos(theta)
                            R[1, 1] = 1.
                            tr_mesh.vertices[...] = \
                                tr_mesh.vertices.dot(R) + furniture.position
                            tr_mesh.vertices[...] = tr_mesh.vertices - s.centroid
                            trimesh_meshes.append(tr_mesh)
                        renderables = renderables_filtered
                        if args.with_windows and s.windows is not None:
                            # Get the walls
                            windows, tr_windows = get_windows(
                                s,
                                [
                                    os.path.join(args.path_to_walls_textures, fi)
                                    for fi in os.listdir(args.path_to_walls_textures)
                                ]
                            )
                            renderables += windows
                            trimesh_meshes += tr_windows
                            trimesh_names += ["window"]

                        if args.with_floor_layout:
                            # Get a floor plan
                            floor_plan, tr_floor = get_floor_plan(
                                s,
                                [
                                    os.path.join(args.path_to_floor_plan_textures, fi)
                                    for fi in os.listdir(args.path_to_floor_plan_textures)
                                ]
                            )
                            renderables += floor_plan
                            trimesh_meshes += tr_floor
                            trimesh_names += ["floor"]

                        if args.with_walls:
                            # Get the walls
                            walls, tr_walls = get_walls(
                                s,
                                [
                                    os.path.join(args.path_to_walls_textures, fi)
                                    for fi in os.listdir(args.path_to_walls_textures)
                                ]
                            )
                            renderables += walls
                            trimesh_meshes += tr_walls
                            trimesh_names += ["wall"]

                        if args.with_doors and s.doors is not None:
                            # Get the doors
                            doors, tr_doors = get_doors(
                                s,
                                [
                                    os.path.join(args.path_to_door_textures, fi)
                                    for fi in os.listdir(args.path_to_door_textures)
                                ]
                            )

                            # Make sure that the mesh that corresponds to the door does
                            # not have holes
                            doors = doors[0]
                            tr_doors = tr_doors[0]
                            # From the trimesh create a rectangular mesh for the door
                            bounding_box = tr_doors.bounding_box
                            m2 = Mesh.from_boxes(
                                bounding_box.centroid[None],
                                (bounding_box.bounds[1] - bounding_box.bounds[0])/ 2,
                                colors=(0.3, 0.3, 0.3)
                            )
                            renderables += [m2]
                            trimesh_meshes += [
                                get_textured_trimesh(
                                    m2,
                                    [
                                        os.path.join(args.path_to_door_textures, fi)
                                        for fi in os.listdir(args.path_to_door_textures)
                                    ]
                                )
                            ]
                            trimesh_names += ["door"]
                        if not args.with_screen:
                            # path_to_image = "{}/{}_".format(args.output_directory, s.uid)
                            # behaviour = SaveFrames(path_to_image+"{:03d}.png", 1)
                            behaviour = SaveFrames(path_to_image, 1)
                            behaviour._i = img_index - 1
                            behaviours += [behaviour]
                            render(
                                renderables,
                                size=args.window_size,
                                camera_position=args.camera_position,
                                camera_target=args.camera_target,
                                up_vector=args.up_vector,
                                background=args.background,
                                behaviours=behaviours,
                                n_frames=1,
                                scene=scene
                            )
                        else:
                            show(
                                renderables,
                                behaviours=behaviours+[SnapshotOnKey()],
                                size=args.window_size,
                                camera_position=args.camera_position,
                                camera_target=args.camera_target,
                                light=args.camera_position,
                                up_vector=args.up_vector,
                                background=args.background,
                            )
                        # print("self._camera post", scene._camera.shape, scene._camera)
                        # print("self._camera_position post", scene._camera_position.shape, scene._camera_position)
                        # print("self._camera_target post", scene._camera_target.shape, scene._camera_target)
                        # print("self.camera_matrix post", scene.camera_matrix.shape, scene.camera_matrix)
                        # print("self.mv post", scene.mv.shape, scene.mv)
                        # print("self.mvp post", scene.mvp.shape, scene.mvp)
                        # Create a trimesh scene and export it
                        # path_to_objs = os.path.join(
                        #     args.output_directory,
                        #     "train_{}".format(args.scene_id)
                        # )
                        # if not os.path.exists(path_to_objs):
                        #     os.mkdir(path_to_objs)
                        # export_scene(path_to_objs, trimesh_meshes)


if __name__ == "__main__":
    main(sys.argv[1:])