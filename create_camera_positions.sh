# Bedroom
path_in=/path/to/data/3dfront/processed/bedrooms_without_lamps/
path_out=/path/to/data/3dfront/processed/bedrooms_without_lamps_full_labels
cd scripts
# python create_camera_positions_trajectory.py --start-idx 0 --end-idx 6000 --path-in $path_in --path-out $path_out --num-samples-scene 40
python create_camera_positions.py --start-idx 0 --end-idx 6000 --path-in $path_in --path-out $path_out --num-samples-scene 40

# Living Rooms
# path_in=/path/to/data/3dfront/processed/living_room_without_lamps/
# path_out=/path/to/data/3dfront/processed/living_room_without_lamps_full_labels
# cd scripts
# python create_camera_positions_living.py --start-idx 0 --end-idx 6000 --path-in $path_in --path-out $path_out --num-samples-scene 40
