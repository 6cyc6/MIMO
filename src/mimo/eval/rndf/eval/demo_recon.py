import os
import os.path as osp

import numpy as np
import polyscope as ps

import torch
import trimesh
from marshmallow_dataclass import dataclass
from mfim.model.vnn_mfim_net import VNNMFIMShared
from robot_utils.pkg.install import DataConfig
from robot_utils.py.utils import load_dataclass

import rndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from rndf_robot.mesh_reconstruction.mesh_reconstruction import MeshReconstruction
from rndf_robot.model.vnn_occ_sdf_scf_net import VNNOccSdfScfNetMulti
from rndf_robot.utils import path_util


@dataclass
class InferenceConfig(DataConfig):
    with_normals: bool = False
    occ: bool = True
    padding: float = 0.1
    threshold: float = 0.7
    resolution0: int = 32
    upsampling_steps: int = 2
    refinement_steps: int = 0
    vis: bool = False


if __name__ == "__main__":
    # demo_path = osp.join(path_util.get_rndf_data(), 'relation_demos/release_demos', "bottle_in_container_relation")
    # demo_path = osp.join(path_util.get_rndf_data(), 'relation_demos/release_demos', "bowl_on_mug_relation")
    demo_path = osp.join(path_util.get_rndf_data(), 'relation_demos/release_demos', "mug_on_rack_relation")

    o_dim = 16

    demo_files = [fn for fn in sorted(os.listdir(demo_path)) if fn.endswith('.npz') and 'recon' not in fn]
    demos = []
    save_names = []
    for f in demo_files:
        demo = np.load(demo_path + '/' + f, allow_pickle=True)
        file_name = f.split('.')[0]
        save_names.append(demo_path + f'/{file_name}_recon')
        demos.append(demo)
    print(demos)

    cfg = load_dataclass(InferenceConfig, 'demo_recon.yaml')

    dev = torch.device('cuda:0')

    # master object
    # model_path = "/ikun/git_project/multi-feature-implicit-model/model_weights/bottle/bottle_multi.pth"
    # model_path = "/ikun/git_project/multi-feature-implicit-model/model_weights/bottle/mfim_bottle_multi.pth"
    # model_path = "/ikun/git_project/multi-feature-implicit-model/model_weights/bowl/bowl_multi.pth"
    # model_path = "/ikun/git_project/multi-feature-implicit-model/model_weights/mug/mug_mfim.pth"
    # model_path = "/ikun/git_project/multi-feature-implicit-model/model_weights/mug/mug_occ.pth"

    # model_path = "/home/ikun/master-thesis/relational_ndf/src/rndf_robot/model_weights/ndf_vnn/rndf_weights/ndf_bottle.pth"

    # mfim4
    model_path = "/home/ikun/master-thesis/ndf_robot/src/ndf_robot/model_weights/mfim_mug_ns16.pth"
    # model_path = "/home/ikun/master-thesis/ndf_robot/src/ndf_robot/model_weights/mfim_bowl_ns16.pth"
    # model_path = "/home/ikun/master-thesis/ndf_robot/src/ndf_robot/model_weights/mfim_bottle_ns16.pth"

    # model_master = VNNOccSdfScfNetMulti(latent_dim=256, model_type='pointnet', o_dim=o_dim, return_features=True, sigmoid=True).cuda()
    model_master = VNNMFIMShared(latent_dim=256, model_type='pointnet', o_dim=16, return_features=True).cuda()

    # model_master = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=True).cuda()
    model_master.load_state_dict(torch.load(model_path))

    print("NDFs is successfully loaded.")
    model_master = model_master.to(dev)
    model_master.eval()
    mesh_recon_master = MeshReconstruction(model=model_master, device=dev, cfg=cfg)

    # slave object
    # model_path = "/ikun/git_project/multi-feature-implicit-model/model_weights/container/container_mfim.pth"
    # model_path = "/ikun/git_project/multi-feature-implicit-model/model_weights/mug/mug_multi.pth"
    # model_path = "/ikun/git_project/multi-feature-implicit-model/model_weights/rack/rack_multi.pth"

    # model_path = "/home/ikun/master-thesis/relational_ndf/src/rndf_robot/model_weights/ndf_vnn/rndf_weights/ndf_container.pth"

    # mfim4
    # model_path = "/home/ikun/master-thesis/ndf_robot/src/ndf_robot/model_weights/mfim_mug_ns16.pth"
    # model_path = "/home/ikun/master-thesis/ndf_robot/src/ndf_robot/model_weights/mfim_bowl_ns16.pth"
    # model_path = "/home/ikun/master-thesis/ndf_robot/src/ndf_robot/model_weights/mfim_bottle_ns16.pth"
    model_path = "/home/ikun/master-thesis/relational_ndf/src/rndf_robot/model_weights/ndf_vnn/mfim_weights/rack_mfim_ns16.pth"
    # model_path = "/home/ikun/master-thesis/relational_ndf/src/rndf_robot/model_weights/ndf_vnn/mfim_weights/container_mfim_ns16.pth"

    # model_slave = VNNOccSdfScfNetMulti(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=True).cuda()
    # model_slave = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=False).cuda()
    model_slave = VNNMFIMShared(latent_dim=256, model_type='pointnet', o_dim=16, return_features=True).cuda()

    model_slave.load_state_dict(torch.load(model_path))

    print("MFIM is successfully loaded.")
    model_slave = model_slave.to(dev)
    model_slave.eval()
    mesh_recon_slave = MeshReconstruction(model=model_slave, device=dev, cfg=cfg)

    for i in range(len(demos)):
        print(i)
        demo = demos[i]
        pcd_demo_master = demo['multi_obj_final_pcd'].item()["child"]
        pcd_demo_slave = demo['multi_obj_final_pcd'].item()["parent"]

        # recon master object
        np.random.shuffle(pcd_demo_master)
        pcd_demo_master = pcd_demo_master[:1500]
        pcd_mean_master = np.mean(pcd_demo_master, axis=0)
        pcd_demo = pcd_demo_master - pcd_mean_master
        mesh_master = mesh_recon_master.from_occupancy(pcd_demo)
        pts = mesh_master.sample(2000)
        pts_master = pts + pcd_mean_master + np.array([0, 0, 1 / 256 * 2])

        # recon slave object
        np.random.shuffle(pcd_demo_slave)
        pcd_demo_slave = pcd_demo_slave[:1500]
        pcd_mean_slave = np.mean(pcd_demo_slave, axis=0)
        pcd_demo = pcd_demo_slave - pcd_mean_slave
        mesh_slave = mesh_recon_slave.from_occupancy(pcd_demo)
        pts = mesh_slave.sample(2000)
        pts_slave = pts + pcd_mean_slave

        # # vis
        # ps.init()
        # ps.set_up_dir("z_up")
        # ps.set_ground_plane_mode('none')
        #
        # ps.register_point_cloud(f"master", pts_master, enabled=True)
        # ps.register_point_cloud(f"slave", pts_slave, enabled=True)
        # ps.register_point_cloud(f"demo_master", pcd_demo_master + np.array([0.5, 0, 0]), enabled=True)
        # ps.register_point_cloud(f"demo_slave", pcd_demo_slave + np.array([0.5, 0, 0]), enabled=True)
        #
        # ps.show()

        # pcl = trimesh.points.PointCloud(pcd_demo_master - pcd_mean_master)
        #
        # scene = trimesh.scene.Scene([mesh_master, pcl])
        # scene.show()

        np.savez(save_names[i],
                 pcd_child=pts_master,
                 pcd_parent=pts_slave)
