# Training R-NDFs

**Download all data assets**

If you want the full dataset (~250GB for 5 object classes):
```
./scripts/download_training_data.bash 
```
If you want just the mug dataset (~50 GB -- other object class data can be downloaded with the according scripts):
```
./scripts/download_mug_training_data.bash 
```

If you want to recreate your own dataset, see Data Generation section

**Run training**
```
cd src/rndf_robot/training
python train_vnn_sdf_net.py --experiment_name bottle_rndf --is_shapenet --obj_class bottle
```

## Notes on training

We utilize the intermediate activations of a pre-trained [DeepSDF network](https://arxiv.org/abs/1901.05103) (with [vector neuron](https://arxiv.org/abs/2104.12229) layers in the point cloud encoder) as our 3D spatial descriptors. Therefore, our training procedure directly mimics the procedure for training an SO(3) equivariant Deep-SDF for 3D reconstruction. Note that the original NDF paper used [occupancy networks](https://arxiv.org/abs/1812.03828), which can also work well for learning a continuous field of spatial descriptors. We found in our R-NDF work that training using signed distance instead of occupancy tends to lead to smoother optimization landscapes. The only change we have made to the training setup from the previous NDF repo is to have the network predict signed-distance instead of occupancy. The rest of the structure in the training code is essentially identical (the details on the training pipeline are repeated here to be self-contained).

The file [`training/dataio.py`](../src/rndf_robot/training/dataio.py) contains ourcustom dataset class. The path of the directory containing the training data is set in the constructor of the dataset class. Each sample contains rendered depth images of the object (we use ShapeNet objects for all our experiments), ground truth occupancy for the object, and camera poses that are used to reconstruct the 3D point cloud from the depth images. More information on the dataset can be found [here](dataset.md).

The rest of the training code is adapted from the training code in the [occupancy network repo](https://github.com/autonomousvision/occupancy_networks). We have found that training the network for 50-100 epochs leads to good performance.

Checkpoints are saved in the folder `model_weights/$LOGGING_ROOT/$EXPERIMENT_NAME/checkpoints`, where `$LOGGING_ROOT` and `$EXPERIMENT_NAME` are set via the training script `argparse` args.  

If you want to make sure training is working correctly without waiting for the full dataset to download, we provide a smaller dataset with the same format that can be quickly downloaded. To use this mini-training set for verifying the pipeline works, run `./scripts/download_mini_training_data.sh` from the root directory (after sourcding `ndf_env.sh`). The downloaded data folders have the same names as our full dataset, but with a much smaller number of individual samples.
