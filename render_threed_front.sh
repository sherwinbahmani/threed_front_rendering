# Bedrooms diverse views
source /path/to/miniconda/etc/profile.d/conda.sh
conda activate blender
cd BlenderProc-main
# Download cc textures
path_cc_textures=/path/to/data/3dfront/processed/blender/cc_textures
# blenderproc run blenderproc/scripts/download_cc_textures.py $path_cc_textures

path_to_3d_front_dataset_dir=/path/to/data/3D-FRONT/
path_to_3d_future_dataset_dir=/path/to/3D-FUTURE-v2/3D-FUTURE/
path_to_3d_future_model_info=/path/to/code/ATISS/demo/model_info.json
path_to_3d_front_texture=/path/to/3D-FRONT-texture
outdir=/path/to/data/3dfront/processed/bedrooms_without_lamps_full_raw/raw
path_labels=/path/to/data/3dfront/processed/bedrooms_without_lamps_full_labels_vertices
outdir_img=/path/to/data/3dfront/processed/bedrooms_without_lamps_full/images
for scene_idx in {0..6000}
do
blenderproc run examples/datasets/front_3d_with_improved_mat_traj_same/main.py $path_to_3d_front_dataset_dir $path_to_3d_future_dataset_dir $path_to_3d_front_texture $path_cc_textures $path_labels $outdir $scene_idx
for frame_idx in {0..39}
do
blenderproc vis hdf5 $outdir --flip=true --keys colors --save $outdir_img --scene_idx $scene_idx --frame_idx $frame_idx --path_labels $path_labels
done
done

# Living room diverse views
source /path/to/miniconda/etc/profile.d/conda.sh
conda activate blender
cd BlenderProc-main
# Download cc textures
path_cc_textures=/path/to/data/3dfront/processed/blender/cc_textures
# blenderproc run blenderproc/scripts/download_cc_textures.py $path_cc_textures

path_to_3d_front_dataset_dir=/path/to/data/3D-FRONT/
path_to_3d_future_dataset_dir=/path/to/3D-FUTURE-v2/3D-FUTURE/
path_to_3d_future_model_info=/path/to/code/ATISS/demo/model_info.json
path_to_3d_front_texture=/path/to/3D-FRONT-texture
outdir=/path/to/data/3dfront/processed/living_room_without_lamps_full_raw/raw
path_labels=/path/to/data/3dfront/processed/living_room_without_lamps_full_labels_vertices
outdir_img=/path/to/data/3dfront/processed/living_room_without_lamps/images
for scene_idx in {0..6000}
do
blenderproc run examples/datasets/front_3d_with_improved_mat_traj_same_living/main.py $path_to_3d_front_dataset_dir $path_to_3d_future_dataset_dir $path_to_3d_front_texture $path_cc_textures $path_labels $outdir $scene_idx
for frame_idx in {0..39}
do
blenderproc vis hdf5 $outdir --flip=true --keys colors --save $outdir_img --scene_idx $scene_idx --frame_idx $frame_idx --path_labels $path_labels
done
done