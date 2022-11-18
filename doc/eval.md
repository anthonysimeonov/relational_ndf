# R-NDF evaluation in PyBullet simulation

**Download all the object data assets**
```
./scripts/download_obj_mesh_data.bash
```

**Download pretrained weights**
```
./scripts/download_demo_weights.bash
```

**Download demonstrations**
```
./scripts/download_relational_demonstrations.bash
```

**Run evaluation**

If you are running this command on a remote machine, be sure to remove the `--pybullet_viz` flag!

```
cd src/rndf_robot/eval
CUDA_VISIBLE_DEVICES=0 \
    python evaluate_relations_multi_ndf.py --parent_class mug --child_class bowl \
    --exp bowl_on_mug_upright_pose_new \
    --parent_model_path ndf_vnn/rndf_weights/ndf_mug.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_bowl.pth \
    --is_parent_shapenet_obj --is_child_shapenet_obj \
    --rel_demo_exp release_demos/bowl_on_mug_relation --pybullet_server \
    --opt_iterations 650 \
    --parent_load_pose_type random_upright --child_load_pose_type random_upright \
    --pybullet_viz
```

## Notes on simulated evaluation setup
We set up an experiment with a pair of objects on a tabletop to evaluate how R-NDF enables localizing coordinate frames near task-relevant parts of the objects and using these estimated frames to perform rearrangement. [`eval/evaluate_relations_multi_ndf.py`](../src/rndf_robot/eval/evaluate_relations_multi_ndf.py) contains the code for running this experiment. The main steps it runs through to set up the experiment are as follows:
- Set up trained neural descriptor field by loading trained network weights
- Load in relational demonstration data and, if necessary, perform the alignment of the query points in the demonstrations using the trained NDFs
- Set up objects to be used in test, making sure that these are objects that were neither included in training nor in the demonstrations.

We then iterate through the test objects. The NDF energy optimization procedure is run on each object point cloud observation and the relative transform between the localized frames is used for transforming the pose of one object such that it satisfies a desired relation with the other object. The rearrangement is executed by resetting the state of the object in the simulator, and the success/failure is tracked to obtain the experimental results. These results are saved in the folder `eval_data` for post-processing.

### Metrics explanation
Our main experimental metric is place success rate. The placement success check works by checking that the transformed object ("object B" or the "child object") is in contact with the static object ("object A" or the "parent object") and that the two objects aren't in collision. For tasks requiring a particular orientation (such as having an upright bottle placed in a container), we also check the final orientation of the child object. Non-collision is checked by turning the pair of objects in their final pose upside down, turning on the physics, and ensuring that the child object falls off the parent object. If it doesn't this is usually because the objects are stuck together, which happens when the inferred transformation places the object in a pose causing large interpenetration. These trials are counted as failures. 

### Commands for running experiments on other categories (bowls, bottles) with our pretrained weights

***Remember to download all the necessary object/demonstration/weight files! See the main [README](../README.md)***

Mug on rack:

```
cd src/rndf_robot/eval
CUDA_VISIBLE_DEVICES=0 \
    python evaluate_relations_multi_ndf.py --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_upright_pose_new \
    --parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
    --is_child_shapenet_obj \
    --rel_demo_exp release_demos/mug_on_rack_relation \
    --pybullet_server \
    --opt_iterations 650 \
    --parent_load_pose_type random_upright --child_load_pose_type random_upright \
    --pybullet_viz
```
To also perform demonstration alignment and create new target descriptors, add ` --new_descriptors --query_scale 0.035 `

Bottle in container:

```
cd src/rndf_robot/eval
CUDA_VISIBLE_DEVICES=0 \
    python evaluate_relations_multi_ndf.py --parent_class syn_container --child_class bottle \
    --exp bottle_in_container_upright_pose_new \
    --parent_model_path ndf_vnn/rndf_weights/ndf_container.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_bottle.pth \
    --is_child_shapenet_obj \
    --rel_demo_exp release_demos/bottle_in_container_relation \
    --pybullet_server \
    --opt_iterations 650 \
    --pc_reference child \
    --parent_load_pose_type random_upright --child_load_pose_type random_upright \
    --pybullet_viz
```
To also perform demonstration alignment and create new target descriptors, add ` --new_descriptors --query_scale 0.025 --pc_reference child `

Bowl on mug:
```
cd src/rndf_robot/eval
CUDA_VISIBLE_DEVICES=0 \
    python evaluate_relations_multi_ndf.py --parent_class mug --child_class bowl \
    --exp bowl_on_mug_upright_pose_new \
    --parent_model_path ndf_vnn/rndf_weights/ndf_mug.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_bowl.pth \
    --is_parent_shapenet_obj --is_child_shapenet_obj \
    --rel_demo_exp release_demos/bowl_on_mug_relation \
    --pybullet_server \
    --opt_iterations 650 \
    --parent_load_pose_type random_upright --child_load_pose_type random_upright \
    --pybullet_viz
```
To also perform demonstration alignment and create new target descriptors, add ` --new_descriptors --query_scale 0.025 `
