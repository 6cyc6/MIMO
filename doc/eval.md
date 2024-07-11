# Evaluation using simulation from NDF and R-NDF  

---

We evaluate our model using the simulation experiments provided by [Neural Descriptor Field (NDF)](https://github.com/anthonysimeonov)
and [Relational NDF](https://github.com/anthonysimeonov/relational_ndf). We modify the code to compare our model with 
NDF, NIFT and R-NDF. For more details about the experiments, please refer to their repos.

## Extra setup
### Install the pkg for solving inverse kinematic: 
```
git clone https://github.com/anthonysimeonov/pybullet-planning.git
cd pybullet-planning/pybullet_tools/ikfast/franka_panda
python setup.py
```

### Download object data assets
```
bash scripts/download_obj_data.sh
```
The object meshes would be put in `mimo/eval/rndf/descriptions/objects`. 

To run the NDF experiment, run the following command to move the object meshes:
```
mv MIMO_DIR/eval/rndf/descriptions/objects MIMO_DIR/eval/ndf/descriptions/objects
```
To move back for R-NDF experiment:
```
mv MIMO_DIR/eval/ndf/descriptions/objects MIMO_DIR/eval/rndf/descriptions/objects
```

### Download pretrained weights
```
bash scripts/download_model_weights_eval.sh
```

### Download demonstrations
```
bash scripts/download_ndf_demo.sh  # for NDF
bash scripts/download_rndf_demo.sh  # for RNDF
```

---

## Run NDF experiment
Remove `--pybullet_viz` for machine without monitor.
Set `--single_view` for using single viewpoint.
Set `--n_demos 1` for using single demonstration. 
Set `--any_pose` for arbitrary pose object initialization.
Set `--recon` to enable mesh reconstruction and point cloud resampling (only works for MIMO).
Set `--ibs` for sampling query points via IBS (check [NIFT](https://github.com/zzilch/NIFT.git) for more details).

Mug:
```
cd [$MIMO_DIR]/eval/ndf
python evaluate_ndf.py \
--demo_exp grasp_rim_hang_handle_gaussian_precise_w_shelf \
--object_class mug \
--opt_iterations 500 \
--only_test_ids \
--rand_mesh_scale \
--model mimo \
--model_path mug_mimo_ns16 \
--save_vis_per_model \
--config eval_mug_gen \
--exp mimo_mug_test \
--pybullet_viz \
--single_view \
--any_pose \
--recon \
--n_demos 1
```

Bowl:
```
cd [$MIMO_DIR]/eval/ndf
python evaluate_ndf.py \
--demo_exp grasp_rim_anywhere_place_shelf_all_methods_multi_instance \
--object_class bowl \
--opt_iterations 500 \
--only_test_ids \
--rand_mesh_scale \
--model mimo \
--model_path bowl_mimo_ns16 \
--save_vis_per_model \
--config eval_bowl_gen \
--exp mimo_bowl_test \
--pybullet_viz \
--single_view \
--any_pose \
--recon \
--n_demos 1
```

Bottle:
```
cd [$MIMO_DIR]/eval/ndf
python evaluate_ndf.py \
--demo_exp grasp_side_place_shelf_start_upright_all_methods_multi_instance \
--object_class bottle \
--opt_iterations 500 \
--only_test_ids \
--rand_mesh_scale \
--model mimo \
--model_path bottle_mimo_ns16 \
--save_vis_per_model \
--config eval_bottle_gen \
--exp mimo_bottle_test \
--pybullet_viz \
--single_view \
--any_pose \
--recon \
--n_demos 1
```

You can also change the configuration for reconstruction in `[$MIMO_DIR]/eval/config/ndf_exp.yaml`.

Please check [Neural Descriptor Field (NDF)](https://github.com/anthonysimeonov) for more details.

---

## Run R-NDF experiment
After running the script below, open a new terminal and run `meshcat-server` to start the simulation.

Mug on Rack:
```
cd [$MIMO_DIR]/eval/rndf
python evaluate_ndf.py \
--parent_class syn_rack_easy \
--child_class mug \
--exp mimo4_mug_on_rack_test \
--parent_model_path rack_mimo_ns16.pth \
--child_model_path mug_mimo_ns16.pth \
--is_child_shapenet_obj \
--rel_demo_exp release_demos/mug_on_rack_relation \
--pybullet_server \
--opt_iterations 600 \
--parent_load_pose_type random_upright \
--child_load_pose_type any_pose \
--pybullet_viz \
--new_descriptors \
--model mimo \ 
--query_scale 0.015 \
--pc_reference child \
--recon \
--single_view \
--n_demos 1 
```

Bowl on Mug:
```
cd [$MIMO_DIR]/eval/rndf
python evaluate_ndf.py \
--parent_class mug \
--child_class bowl \
--exp mimo4_bowl_on_mug_test \
--parent_model_path mug_mimo_ns16.pth \
--child_model_path bowl_mimo_ns16.pth \
--is_child_shapenet_obj \
--is_parent_shapenet_obj \
--rel_demo_exp release_demos/bowl_on_mug_relation \
--pybullet_server \
--opt_iterations 600 \
--parent_load_pose_type random_upright \
--child_load_pose_type any_pose \
--pybullet_viz \
--new_descriptors \
--model mimo \ 
--query_scale 0.015 \
--pc_reference child \
--recon \
--single_view \
--n_demos 1 
```

Bottle in Container:
```
cd [$MIMO_DIR]/eval/rndf
python evaluate_ndf.py \
--parent_class syn_container \
--child_class bottle \
--exp mimo4_bottle_in_container_test \
--parent_model_path container_mimo_ns16.pth \
--child_model_path bottle_mimo_ns16.pth \
--is_child_shapenet_obj \
--rel_demo_exp release_demos/bottle_in_container_relation \
--pybullet_server \
--opt_iterations 600 \
--parent_load_pose_type random_upright \
--child_load_pose_type any_pose \
--pybullet_viz \
--new_descriptors \
--model mimo \ 
--query_scale 0.015 \
--pc_reference child \
--recon \
--single_view \
--n_demos 1 
```

Please check [Relational NDF](https://github.com/anthonysimeonov/relational_ndf) for more details.