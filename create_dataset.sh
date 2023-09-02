# 1. Pre-process dataset following ATISS
path_to_output_dir=/path/to/data/3dfront/processed/bedrooms_without_lamps
path_to_3d_front_dataset_dir=/path/to/data/3D-FRONT
path_to_3d_future_dataset_dir=/path/to/data/3D-FUTURE-model
path_to_3d_future_model_info=/path/to/code/ATISS/demo/model_info.json
path_to_floor_plan_texture_images=/path/to/code/ATISS/demo/floor_plan_texture_images
cd scripts
xvfb-run -a python preprocess_data.py $path_to_output_dir $path_to_3d_front_dataset_dir $path_to_3d_future_dataset_dir $path_to_3d_future_model_info $path_to_floor_plan_texture_images --dataset_filtering threed_front_bedroom --without_lamps

# 2. Create camera coordinates
bash create_camera_positions.sh

# 3. Normalize labels for rendering
bash create_norm_labels.sh

# 4. Render the dataset
bash render_threed_front.sh
