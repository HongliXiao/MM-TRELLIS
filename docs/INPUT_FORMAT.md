# Input format

`inference.py` expects two inputs for each vehicle instance:

1. one or more cropped/segmented RGB vehicle images;
2. one preprocessed guidance point cloud, usually aggregated from LiDAR points and optionally symmetrized / DBSCAN-filtered.

## Default directory layout

```text
examples_input/
├── initial_test_instances/
│   └── static_006_002/
│       ├── test/*.png
│       └── validation/*.png
└── drivestudio_way_processed/
    └── 006/
        └── static_006_002/
            └── aggregated_instance_lidar_pts/
                ├── 1.filtered-bottom_threshold0.075.ply
                ├── 2.symmetric-DBSCAN.ply
                └── 2.symmetric-DBSCAN.png
```

For `instance = static_006_002`, the script extracts `scene_id = 006` and loads:

```text
<pts-root>/006/static_006_002/aggregated_instance_lidar_pts/2.symmetric-DBSCAN.ply
```

Use `--point_cloud_name` to select another file in `aggregated_instance_lidar_pts/`.

## Image input

Images are loaded from:

```text
<input-root>/<instance>/<split>/*.{png,jpg,jpeg,webp}
```

The default split is `test`.

## Point-cloud input

Point clouds should be PLY files readable by Open3D. The script provides three normalization modes:

- `scale-only`: scales the point cloud by its maximum side length but preserves its center. This matches the original experiment script.
- `unit-box`: recenters the point cloud and scales it into a unit cube.
- `none`: uses point coordinates as-is.

Example:

```bash
python inference.py \
  --point_normalization scale-only \
  --point_normalization_scale 0.98
```

## Adding your own data

For a new instance, prepare:

```text
my_images_root/static_001_000/test/*.png
my_points_root/001/static_001_000/aggregated_instance_lidar_pts/2.symmetric-DBSCAN.ply
```

Then run:

```bash
python inference.py \
  --input_root my_images_root \
  --pts_root my_points_root \
  --instances static_001_000 \
  --split test \
  --output_dir outputs/my_instance
```

If your naming convention is different from `static_<scene>_<track>`, edit `default_point_cloud_path()` in `inference.py` or add a metadata file mapping instance ids to point-cloud paths.
