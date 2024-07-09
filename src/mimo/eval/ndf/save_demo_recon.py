import os, os.path as osp
import random
import numpy as np
import time
import signal

import pyibs
import torch
import argparse
import shutil

import polyscope as ps
import pybullet as p
import trimesh
from mfim.eval.config.eval_cfg import InferenceConfig
from mfim.model.vnn_mfim_net import VNNMFIMShared

from scipy.spatial import KDTree
from scipy.interpolate import RBFInterpolator

from airobot import Robot
from airobot import log_info, log_warn, log_debug, log_critical, set_log_level
from airobot.utils import common
from airobot import log_info
from airobot.utils.common import euler2quat

from mfim.reconstruction.mesh_reconstruction import MeshReconstruction
from mfim.utils.config_util import load_dataclass

import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.eval.utils import matrix_from_pose
from ndf_robot.model.vnn_occ_sdf_scf_net import VNNOccSdfScfNetMulti
from ndf_robot.utils import util, trimesh_util
from ndf_robot.utils.util import np2img

from ndf_robot.opt.optimizer import OccNetOptimizer
from ndf_robot.robot.multicam import MultiCams
from ndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from ndf_robot.config.default_obj_cfg import get_obj_cfg_defaults
from ndf_robot.utils import path_util
from ndf_robot.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list
from ndf_robot.utils.franka_ik import FrankaIK
from ndf_robot.utils.eval_gen_utils import (
    soft_grasp_close, constraint_grasp_close, constraint_obj_world, constraint_grasp_open,
    safeCollisionFilterPair, object_is_still_grasped, get_ee_offset, post_process_grasp_point,
    process_demo_data_rack, process_demo_data_shelf, process_xq_data, process_xq_rs_data, safeRemoveConstraint,
)

# config
n_demos = 0

# obj_class = "mug"
# num_demo = 12
# shelf = False

obj_class = "bowl"
num_demo = 22
shelf = True

# obj_class = "bottle"
# num_demo = 20
# shelf = False

shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(), obj_class + '_centered_obj_normalized')

# demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', obj_class, "grasp_rim_hang_handle_gaussian_precise_w_shelf")
demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', obj_class, "grasp_rim_anywhere_place_shelf_all_methods_multi_instance")
# demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', obj_class, "grasp_side_place_shelf_start_upright_all_methods_multi_instance")

cfg = load_dataclass(InferenceConfig, '/ikun/projects/multi-feature-implicit-model/src/mfim/eval/config/ndf_exp.yaml')

# load model
# model = VNNOccSdfScfNetMulti(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=True).cuda()
model = VNNMFIMShared(latent_dim=256, model_type='pointnet', o_dim=16, return_features=True).cuda()

# checkpoint_path = "/home/ikun/master-thesis/ndf_robot/src/ndf_robot/model_weights/mfim_mug_multi.pth"
# checkpoint_path = "/home/ikun/master-thesis/ndf_robot/src/ndf_robot/model_weights/mfim_bowl_multi.pth"
# checkpoint_path = "/home/ikun/master-thesis/ndf_robot/src/ndf_robot/model_weights/mfim_bottle_multi.pth"

# checkpoint_path = "/home/ikun/master-thesis/ndf_robot/src/ndf_robot/model_weights/mfim_mug_ns16.pth"
checkpoint_path = "/home/ikun/master-thesis/ndf_robot/src/ndf_robot/model_weights/mfim_bowl_ns16.pth"
# checkpoint_path = "/home/ikun/master-thesis/ndf_robot/src/ndf_robot/model_weights/mfim_bottle_ns16.pth"
model.load_state_dict(torch.load(checkpoint_path))

recon = MeshReconstruction(model=model, device='cuda:0', cfg=cfg)

cfg = get_eval_cfg_defaults()
config_fname = osp.join(path_util.get_ndf_config(), 'eval_cfgs', f"eval_{obj_class}_gen" + '.yaml')
if osp.exists(config_fname):
    cfg.merge_from_file(config_fname)
else:
    log_info('Config file %s does not exist, using defaults' % config_fname)
cfg.freeze()

# get filenames of all the demo files
demo_filenames = os.listdir(demo_load_dir)
assert len(demo_filenames), 'No demonstrations found in path: %s!' % demo_load_dir

# strip the filenames to properly pair up each demo file
grasp_demo_filenames_orig = [osp.join(demo_load_dir, fn) for fn in demo_filenames if
                             'grasp_demo' in fn and 'ibs' not in fn]  # use the grasp names as a reference

place_demo_filenames = []
grasp_demo_filenames = []
for i, fname in enumerate(grasp_demo_filenames_orig):
    shapenet_id_npz = fname.split('/')[-1].split('grasp_demo_')[-1]
    place_fname = osp.join('/'.join(fname.split('/')[:-1]), 'place_demo_' + shapenet_id_npz)
    if osp.exists(place_fname):
        grasp_demo_filenames.append(fname)
        place_demo_filenames.append(place_fname)
    else:
        log_warn('Could not find corresponding placement demo: %s, skipping ' % place_fname)

demo_shapenet_ids = []

# get info from all demonstrations
demo_target_info_list = []
demo_rack_target_info_list = []

if n_demos > 0:
    gp_fns = list(zip(grasp_demo_filenames, place_demo_filenames))
    gp_fns = random.sample(gp_fns, n_demos)
    grasp_demo_filenames, place_demo_filenames = zip(*gp_fns)
    grasp_demo_filenames, place_demo_filenames = list(grasp_demo_filenames), list(place_demo_filenames)
    log_warn('USING ONLY %d DEMONSTRATIONS' % len(grasp_demo_filenames))
    print(grasp_demo_filenames, place_demo_filenames)
else:
    log_warn('USING ALL %d DEMONSTRATIONS' % len(grasp_demo_filenames))

grasp_demo_filenames = grasp_demo_filenames[:num_demo]
place_demo_filenames = place_demo_filenames[:num_demo]


max_bb_volume = 0
place_xq_demo_idx = 0
grasp_data_list = []
place_data_list = []
demo_rel_mat_list = []
demo_mesh_path_list = []
grasp_ibs_list = []
place_ibs_list = []

# load all the demo data and look at objects to help decide on query points
for i, fname in enumerate(grasp_demo_filenames):
    # if i == 2:
    #     break
    print('Loading demo from fname: %s' % fname)
    grasp_demo_fn = grasp_demo_filenames[i]
    place_demo_fn = place_demo_filenames[i]
    grasp_data = np.load(grasp_demo_fn, allow_pickle=True)
    place_data = np.load(place_demo_fn, allow_pickle=True)

    grasp_data_list.append(grasp_data)
    place_data_list.append(place_data)

    start_ee_pose = grasp_data['ee_pose_world'].tolist()
    end_ee_pose = place_data['ee_pose_world'].tolist()
    place_rel_mat = util.get_transform(
        pose_frame_target=util.list2pose_stamped(end_ee_pose),
        pose_frame_source=util.list2pose_stamped(start_ee_pose)
    )
    place_rel_mat = util.matrix_from_pose(place_rel_mat)
    demo_rel_mat_list.append(place_rel_mat)

    if i == 0:
        optimizer_gripper_pts, rack_optimizer_gripper_pts, shelf_optimizer_gripper_pts = process_xq_data(grasp_data, place_data, shelf=shelf)
        optimizer_gripper_pts_rs, rack_optimizer_gripper_pts_rs, shelf_optimizer_gripper_pts_rs = process_xq_rs_data(grasp_data, place_data, shelf=shelf)

        if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
            print('Using shelf points')
            place_optimizer_pts = shelf_optimizer_gripper_pts
            place_optimizer_pts_rs = shelf_optimizer_gripper_pts_rs
        else:
            print('Using rack points')
            place_optimizer_pts = rack_optimizer_gripper_pts
            place_optimizer_pts_rs = rack_optimizer_gripper_pts_rs

    if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
        target_info, rack_target_info, shapenet_id = process_demo_data_shelf(grasp_data, place_data, cfg=None, obj_class=obj_class)
    else:
        target_info, rack_target_info, shapenet_id = process_demo_data_rack(grasp_data, place_data, cfg=None, obj_class=obj_class)

    if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
        rack_target_info['demo_query_pts'] = place_optimizer_pts
    demo_target_info_list.append(target_info)
    demo_rack_target_info_list.append(rack_target_info)
    demo_shapenet_ids.append(shapenet_id)
    demo_mesh_path_list.append(osp.join(path_util.get_ndf_obj_descriptions(), f"{obj_class}_centered_obj",
                                        shapenet_id, "models/model_128_df.obj"))

    # grasp
    demo_shape_pts_world = target_info['demo_obj_pts']

    # reconstruction
    np.random.shuffle(demo_shape_pts_world)
    demo_shape_pts_world = demo_shape_pts_world[:1500]
    pcd_mean = np.mean(demo_shape_pts_world, axis=0)
    demo_shape_pts_world -= pcd_mean
    inliers = np.where(np.linalg.norm(demo_shape_pts_world, 2, 1) < 0.22)[0]
    demo_shape_pts_world = demo_shape_pts_world[inliers]
    recon_mesh = recon.from_occupancy(pcd=demo_shape_pts_world)
    recon_pcd = recon_mesh.sample(2000)

    # ps.init()
    # ps.set_up_dir("z_up")
    #
    # ps.register_point_cloud("obj", demo_shape_pts_world, radius=0.005, color=[0, 0, 1], enabled=True)
    # ps.register_point_cloud("recon", recon_pcd, radius=0.005, color=[0, 1, 0], enabled=True)
    #
    # ps.show()

    recon_pcd_world = recon_pcd + pcd_mean
    demo_shape_pts_world += pcd_mean

    # # load ibs
    # grasp_ibs_fn = grasp_demo_fn.split('.')[0] + '_ibs.npz'
    # grasp_ibs_data = np.load(grasp_ibs_fn, allow_pickle=True)
    # gripper_ibs_pts = grasp_ibs_data["ibs"]
    # ones = np.ones((gripper_ibs_pts.shape[0], 1))
    # pts_ones = np.hstack([gripper_ibs_pts, ones])
    # trans_ibs_pts = grasp_ibs_data["trans_mat"] @ pts_ones.T
    # gripper_ibs_pts = trans_ibs_pts[:3, :].T
    #
    # grasp_ibs_list.append(gripper_ibs_pts)
    #
    # place_ibs_fn = place_demo_fn.split('.')[0] + '_ibs.npz'
    # place_ibs_data = np.load(place_ibs_fn, allow_pickle=True)
    # place_ibs_pts = place_ibs_data["ibs"]
    # place_ibs_list.append(place_ibs_pts)

    # ibs save
    # # grasp
    # demo_ee_pose_world = target_info['demo_ee_pose_world']
    # trans_mat = util.matrix_from_pose(util.list2pose_stamped(demo_ee_pose_world))
    # mesh_gripper = trimesh.load_mesh('hand.obj')
    # mesh_gripper.apply_transform(trans_mat)
    # pts_mesh = mesh_gripper.sample(5000)
    #
    # ibs = pyibs.IBS(pts_mesh, demo_shape_pts_world, n=1000)
    # # ibs = pyibs.IBS(pts_mesh, recon_pcd_world, n=1000)
    # pts = ibs.sample_points()
    # # save_file = grasp_demo_fn.split('.')[0] + '_ibs'
    # save_file = grasp_demo_fn.split('.')[0] + '_recon_ibs'
    # # np.savez(save_file, ibs=pts, trans_mat=np.linalg.inv(trans_mat))
    # np.savez(save_file, ibs=pts, trans_mat=np.linalg.inv(trans_mat), recon=recon_pcd_world)

    # place
    # shelf
    shelf_path = "/home/ikun/master-thesis/ndf_robot/src/ndf_robot/descriptions/hanging/table/shelf_back.obj"
    mesh_shelf = trimesh.load_mesh(shelf_path)
    shelf_mat = util.matrix_from_pose(util.list2pose_stamped([0.3875, -0.45, 0.9625, -0.0, -0.0, 0.00039816338692783, 0.9999999207329555]))
    mesh_shelf.apply_transform(shelf_mat)
    demo_pts = mesh_shelf.sample(5000)

    demo_shape_pts_world = rack_target_info['demo_obj_pts']
    demo_shape_pts_world += np.array([0, 0, 0.125])

    # # rack
    # demo_pts = rack_target_info['demo_query_pts_real_shape']
    #
    # # reconstruction
    # np.random.shuffle(demo_shape_pts_world)
    # demo_shape_pts_world = demo_shape_pts_world[:1500]
    # pcd_mean = np.mean(demo_shape_pts_world, axis=0)
    # demo_shape_pts_world -= pcd_mean
    # inliers = np.where(np.linalg.norm(demo_shape_pts_world, 2, 1) < 0.22)[0]
    # demo_shape_pts_world = demo_shape_pts_world[inliers]
    # recon_mesh = recon.from_occupancy(pcd=demo_shape_pts_world)
    # recon_pcd = recon_mesh.sample(2000)
    # # # ps.init()
    # # # ps.set_up_dir("z_up")
    # # #
    # # # ps.register_point_cloud("obj", demo_shape_pts_world, radius=0.005, color=[0, 0, 1], enabled=True)
    # # # ps.register_point_cloud("recon", recon_pcd, radius=0.005, color=[0, 1, 0], enabled=True)
    # # #
    # # # ps.show()
    # #
    # recon_pcd_world = recon_pcd + pcd_mean
    # demo_shape_pts_world = recon_pcd_world
    #
    # ibs = pyibs.IBS(demo_pts, demo_shape_pts_world, n=1000)
    # pts = ibs.sample_points()
    # # save_file = place_demo_fn.split('.')[0] + '_ibs'
    # save_file = place_demo_fn.split('.')[0] + '_recon_ibs'
    # # np.savez(save_file, ibs=pts)
    # np.savez(save_file, ibs=pts, recon=demo_shape_pts_world)

    # vis
    ps.init()
    ps.set_up_dir("z_up")

    ps.register_point_cloud("obj", demo_shape_pts_world, radius=0.005, color=[0, 0, 1], enabled=True)
    ps.register_point_cloud("recon", recon_pcd_world, radius=0.005, color=[0, 1, 0], enabled=True)
    ps.register_point_cloud("rack", demo_pts, radius=0.005, color=[1, 0, 1], enabled=True)
    # ps.register_point_cloud("gripper", pts_mesh, radius=0.005, color=[1, 0, 0], enabled=True)
    # ps.register_point_cloud("ibs", pts, radius=0.005, color=[1, 0, 0], enabled=True)
    # ps.register_point_cloud("opt", shelf_optimizer_gripper_pts, radius=0.005, color=[0, 1, 0], enabled=True)
    ps.show()

# postprocess ibs pts
grasp_ibs_pts = np.concatenate(grasp_ibs_list[:])
tree = KDTree(grasp_ibs_pts)
ds, _ = tree.query(grasp_ibs_pts, k=num_demo + 1, workers=5)
avg_ds = 1 / (ds.sum(axis=1) / num_demo + 1e-10)
pts_weights = avg_ds / avg_ds.sum()
print(pts_weights.sum())
idx = np.arange(grasp_ibs_pts.shape[0])
idx_pts = np.random.choice(idx, 300, p=pts_weights, replace=False)
optimizer_gripper_ibs_pts = grasp_ibs_pts[idx_pts]

place_ibs_pts = np.concatenate(place_ibs_list[:])
tree = KDTree(place_ibs_pts)
ds, _ = tree.query(grasp_ibs_pts, k=num_demo + 1, workers=5)
avg_ds = 1 / (ds.sum(axis=1) / num_demo + 1e-10)
pts_weights = avg_ds / avg_ds.sum()
idx = np.arange(place_ibs_pts.shape[0])
idx_pts = np.random.choice(idx, 300, p=pts_weights, replace=False)
place_optimizer_ibs_pts = place_ibs_pts[idx_pts]

# vis
mesh_gripper = trimesh.load_mesh('gripper.obj')
pts_mesh_gripper = mesh_gripper.sample(5000)

ps.init()
ps.set_up_dir("z_up")

# ps.register_point_cloud("gripper_ibs", optimizer_gripper_ibs_pts, radius=0.005, color=[0, 0, 1], enabled=True)
# ps.register_point_cloud("gripper", pts_mesh_gripper, radius=0.005, color=[1, 1, 1], enabled=True)
ps.register_point_cloud("place", place_optimizer_ibs_pts, radius=0.005, color=[0, 1, 0], enabled=True)
ps.register_point_cloud("rack", demo_pts, radius=0.005, color=[1, 0, 1], enabled=True)
ps.register_point_cloud("pts", place_optimizer_pts, radius=0.005, color=[1, 0, 1], enabled=True)

ps.show()
