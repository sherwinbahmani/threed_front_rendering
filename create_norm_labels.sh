cd scripts

in_dir=/path/to/data/3dfront/processed/bedrooms_without_lamps_full_labels
out_dir=/path/to/data/3dfront/processed/bedrooms_without_lamps_full_labels_vertices
out_dir_norm=/path/to/data/3dfront/processed/bedrooms_without_lamps_full/labels
python add_vertices_calc.py --in-dir $in_dir --out-dir $out_dir
python normalize_dataset.py --in-dir $out_dir --out-dir $out_dir_norm