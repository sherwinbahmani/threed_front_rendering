"""Script used for parsing 3D_FRONT scenes"""
import argparse
import logging
import os
import sys
from typing import List

import numpy as np
from PIL import Image
import pyrr
import trimesh
from scene_synthesis.datasets.threed_front import ThreedFront

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
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-2.0,-2.0,-2.0",
        help="Camer position in the scene"
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
    args = parser.parse_args(argv)
    print("args.camera_position", args.camera_position, "args.camera_target", args.camera_target)
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size)
    if args.with_orthographic_projection:
        scene.camera_matrix = pyrr.Matrix44.orthogonal_projection(
            left=-3.1, right=3.1, bottom=-3.1, top=3.1, near=0.1, far=1000
        )
    scene.light = args.camera_position
    behaviours = []

    d = ThreedFront.from_dataset_directory(
        args.path_to_3d_front_dataset_directory,
        args.path_to_model_info,
        args.path_to_3d_future_dataset_directory,
        path_to_room_masks_dir=None,
        path_to_bounds=None,
        filter_fn=lambda s: s
    )
    print("Loading dataset with {} rooms".format(len(d)))

    for s in d.scenes:
        if s.scene_id == args.scene_id or args.scene_id is None:
            path_to_file = os.path.join(
                args.output_directory, s.scene_id
            )
            # Check optimistically if the file already exists
            if os.path.exists(path_to_file):
                continue
            ensure_parent_directory_exists(path_to_file)

            # Make sure we are the only ones creating this file
            with DirLock(path_to_file + ".lock") as lock:
                if not lock.is_acquired:
                    continue
                if os.path.exists(path_to_file):
                    continue
                renderables = s.furniture_renderables(
                    with_floor_plan_offset=True, with_texture=args.with_texture
                )
                trimesh_meshes = []
                trimesh_names = []
                models_class_labels = []
                for i, furniture in enumerate(s.bboxes):
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
                    path_to_image = "{}/{}_".format(args.output_directory, s.uid)
                    behaviours += [SaveFrames(path_to_image+"{:03d}.png", 1)]
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
                # Create a trimesh scene and export it
                path_to_objs = os.path.join(
                    args.output_directory,
                    "train_{}".format(args.scene_id)
                )
                if not os.path.exists(path_to_objs):
                    os.mkdir(path_to_objs)
                export_scene(path_to_objs, trimesh_meshes)


if __name__ == "__main__":
    main(sys.argv[1:])