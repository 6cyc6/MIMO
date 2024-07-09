import os, os.path as osp
import random
import click

import numpy as np
import polyscope as ps
import trimesh

from mfim.eval.ndf.config.default_eval_cfg import get_eval_cfg_defaults
from mfim.eval.ndf.utils import path_util, util
from mfim.eval.ndf.utils.eval_gen_utils import process_demo_data_shelf, process_xq_rs_data, process_xq_data, \
    process_demo_data_rack

from airobot import log_info, log_warn


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--n_demos",   "-n", type=int,       default=0,   help="num of demos")
@click.option("--obj",   "-o", type=str,       default=None,   help="object type")
@click.option("--ibs",   "-i", type=bool,      default=False,   help="if use ibs")
@click.option("--recon",   "-r", type=bool,      default=False,   help="if use reconstruction")
def main(n_demos, obj, ibs, recon):
    # config
    n_demos = n_demos

    shelf = True
    obj_class = obj
    if obj_class == "mug":
        num_demo = 12
        shelf = False
    elif obj_class == "bowl":
        num_demo = 22
    else:
        num_demo = 20

    if obj_class == "mug":
        demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', obj_class, "grasp_rim_hang_handle_gaussian_precise_w_shelf")
    elif obj_class == "bowl":
        demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', obj_class, "grasp_rim_anywhere_place_shelf_all_methods_multi_instance")
    else:
        demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', obj_class,
                                 "grasp_side_place_shelf_start_upright_all_methods_multi_instance")

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
    grasp_demo_ibs_filenames = []
    place_demo_ibs_filenames = []
    for i, fname in enumerate(grasp_demo_filenames_orig):
        shapenet_id_npz = fname.split('/')[-1].split('grasp_demo_')[-1]
        place_fname = osp.join('/'.join(fname.split('/')[:-1]), 'place_demo_' + shapenet_id_npz)
        if osp.exists(place_fname):
            grasp_demo_filenames.append(fname)
            place_demo_filenames.append(place_fname)
            if recon:
                grasp_ibs_file_name = fname.split('.')[0] + '_recon_ibs.npz'
                grasp_demo_ibs_filenames.append(grasp_ibs_file_name)
                place_ibs_file_name = place_fname.split('.')[0] + '_recon_ibs.npz'
                place_demo_ibs_filenames.append(place_ibs_file_name)
            if not recon and ibs:
                grasp_ibs_file_name = fname.split('.')[0] + '_ibs.npz'
                grasp_demo_ibs_filenames.append(grasp_ibs_file_name)
                place_ibs_file_name = place_fname.split('.')[0] + '_ibs.npz'
                place_demo_ibs_filenames.append(place_ibs_file_name)
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

    # load all the demo data and look at objects to help decide on query points
    for i, fname in enumerate(grasp_demo_filenames):
        print('Loading demo from fname: %s' % fname)
        grasp_demo_fn = grasp_demo_filenames[i]
        place_demo_fn = place_demo_filenames[i]

        grasp_data = np.load(grasp_demo_fn, allow_pickle=True)
        place_data = np.load(place_demo_fn, allow_pickle=True)

        if i == 0:
            optimizer_gripper_pts, rack_optimizer_gripper_pts, shelf_optimizer_gripper_pts = process_xq_data(grasp_data,
                                                                                                             place_data,
                                                                                                             shelf=shelf)
            optimizer_gripper_pts_rs, rack_optimizer_gripper_pts_rs, shelf_optimizer_gripper_pts_rs = process_xq_rs_data(
                grasp_data, place_data, shelf=shelf)

            if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
                print('Using shelf points')
                place_optimizer_pts = shelf_optimizer_gripper_pts
                place_optimizer_pts_rs = shelf_optimizer_gripper_pts_rs
            else:
                print('Using rack points')
                place_optimizer_pts = rack_optimizer_gripper_pts
                place_optimizer_pts_rs = rack_optimizer_gripper_pts_rs

        if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
            target_info, rack_target_info, shapenet_id = process_demo_data_shelf(grasp_data, place_data, cfg=None,
                                                                                 obj_class=obj_class)
        else:
            target_info, rack_target_info, shapenet_id = process_demo_data_rack(grasp_data, place_data, cfg=None,
                                                                                obj_class=obj_class)

        if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
            rack_target_info['demo_query_pts'] = place_optimizer_pts

        pcd_obj_start = target_info['demo_obj_pts']
        pcd_obj_end = rack_target_info['demo_obj_pts']

        demo_ee_pose_world = target_info['demo_ee_pose_world']
        trans_mat = util.matrix_from_pose(util.list2pose_stamped(demo_ee_pose_world))
        mesh_gripper = trimesh.load_mesh('hand.obj')
        mesh_gripper.apply_transform(trans_mat)
        pts_mesh = mesh_gripper.sample(5000)

        # vis
        ps.init()
        ps.set_up_dir("z_up")

        ps.register_point_cloud("obj_s", pcd_obj_start, radius=0.005, color=[0, 0, 1], enabled=True)
        ps.register_point_cloud("obj_e", pcd_obj_end, radius=0.005, color=[0, 0, 1], enabled=True)
        ps.register_point_cloud("gripper", pts_mesh, radius=0.005, color=[1, 0, 0], enabled=True)
        # ps.register_point_cloud("recon", recon_pcd_world, radius=0.005, color=[0, 1, 0], enabled=True)
        # ps.register_point_cloud("demo", demo_pts, radius=0.005, color=[1, 0, 1], enabled=True)
        # ps.register_point_cloud("query", place_optimizer_pts, radius=0.005, color=[0, 1, 0], enabled=True)
        ps.register_point_cloud("real shape", place_optimizer_pts_rs, radius=0.008, color=[0, 0, 0], enabled=True)

        # load ibs
        if recon or ibs:
            grasp_demo_ibs_fn = grasp_demo_ibs_filenames[i]
            place_demo_ibs_fn = place_demo_ibs_filenames[i]
            grasp_ibs_data = np.load(grasp_demo_ibs_fn, allow_pickle=True)
            place_ibs_data = np.load(place_demo_ibs_fn, allow_pickle=True)
            grasp_ibs = grasp_ibs_data["ibs"]
            place_ibs = place_ibs_data["ibs"]
            ps.register_point_cloud("grasp_ibs", grasp_ibs, radius=0.003, color=[1, 1, 0], enabled=True)
            ps.register_point_cloud("place_ibs", place_ibs, radius=0.003, color=[1, 1, 0], enabled=True)
            if recon:
                grasp_recon = grasp_ibs_data["recon"]
                place_recon = place_ibs_data["recon"]
                ps.register_point_cloud("grasp_recon", grasp_recon, radius=0.005, color=[0, 1, 1], enabled=True)
                ps.register_point_cloud("place_recon", place_recon, radius=0.005, color=[0, 1, 1], enabled=True)

        ps.show()


if __name__ == '__main__':
    main()
