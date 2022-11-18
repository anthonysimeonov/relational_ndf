# R-NDF Dataset and Synthetic Objects
**Download all the object .obj assets and ground truth SDF samples**
```
./scripts/download_obj_data.bash
```


**Run data generation**

ShapeNet objects
```
cd src/rndf_robot/data_gen
python shapenet_pcd_gen.py \
    --total_samples 100 \
    --object_class mug \
    --save_dir test_mug \
    --rand_scale \
    --num_workers 2
```

Non-shapenet objects (i.e., that are manually synthesized, such as our synthetic racks or containers)

```
# racks:
cd src/rndf_robot/data_gen
python syn_obj_pcd_gen.py \
    --object_class syn_rack_easy \
    --save_dir test_syn_rack \
    --total_samples 100 \
    --obj_file_dir syn_racks_easy_obj_unnormalized \
    --config syn_rack_easy_data_gen.yaml \
    --rand_scale \
    --num_workers 2

# containers:
cd src/rndf_robot/data_gen
python syn_obj_pcd_gen.py \
    --object_class syn_container \
    --save_dir test_syn_container \
    --total_samples 100 \
    --obj_file_dir box_containers_unnormalized \
    --config container_data_gen.yaml \
    --rand_scale \
    --num_workers 2

```

**Generate your own ground truth signed-distance values**

ShapeNet objects
```
python prepare_sdf_from_obj.py \
    --dataset_name test_mug_sdf \
    --models_directory mug_centered_obj_normalized \
    --is_shapenet

```

Non-shapenet objects
```
python prepare_sdf_from_obj.py \
    --dataset_name test_syn_rack_sdf \
    --models_directory syn_racks_easy_obj_unnormalized 
```

**Generate your own synthetic container/rack .obj files**

(and, optionally, ground truth occupancy)
```
# racks:
cd src/rndf_robot/data_gen/model_gen
python generate_syn_racks.py \
    --obj_name test_syn_rack \
    --mesh_save_dir test_syn_racks \
    --occ_save_dir test_syn_racks_occ  # include --save_occ to also save normalized files and occupancy

# containers:
cd src/rndf_robot/data_gen/model_gen
python generate_syn_containers.py \
    --obj_name test_syn_container \
    --mesh_save_dir test_syn_container \
    --occ_save_dir test_syn_container_occ 
```

## Notes on dataset and dataset generation
### Generating 3D point clouds and ground truth object poses
Our dataset is primarily composed pairs of object point clouds and ground truth signed-distance values of a large number of points sampled in the volume near the shape. These are obtained by running the script at [`data_gen/shapenet_pcd_gen.py`](../src/rndf_robot/data_gen/shapenet_pcd_gen.py) for ShapeNet objects, or [`data_gen/syn_obj_pcd_gen.py`](../src/rndf_robot/data_gen/syn_obj_pcd_gen.py) for synthetic objects, after downloading the object models (see [main README](../README.md)).

The scripts run by placing the object meshes in a PyBullet simulation and rendering depth images using simulated cameras at different poses. The objects are randomly scaled and posed to create more diversity in the dataset to help the point cloud encoder generalize. The points at which the occupancy is evaluated are similarly scaled/transformed based on how the object is adjusted when we load it into the simulator. 

We also have the option of adding other random shapes into the simulator so that the shape is partially occluded in some of the samples (use the `--occlude` flag with the above command).

### Generating ground truth signed distance
We included a script at [`data_gen/prepare_sdf_from_obj.py`](../src/rndf_robot/data_gen/prepare_sdf_from_obj.py) which can generate ground truth signed distance values for a set of query points sampled in the vicinity of an object, using an `.obj` file representing the object mesh. This script was built off of a tool provided in the [shapegan repo](https://github.com/marian42/shapegan). 

### Creating your own synthetic racks and containers
We also included the tools used to create our synthetic racks and containers. These can be found in the scripts at [`data_gen/model_gen/generate_syn_racks.py`](../src/rndf_robot/data_gen/model_gen/generate_syn_racks.py) and [`data_gen/model_gen/generate_syn_containers.py`](../src/rndf_robot/data_gen/model_gen/generate_syn_containers.py). They each have a manually designed way to parametrically generate parts that make up the objects with different dimensions, and then compose these parts into an overall mesh.
