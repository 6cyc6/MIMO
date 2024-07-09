import copy
import os, os.path as osp
import torch
# import pyibs
import polyscope as ps
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import open3d as o3d
import trimesh

from airobot import log_info, log_warn, log_debug, log_critical
from mimo.eval.config.eval_cfg import InferenceConfig

from mimo.utils.config_util import load_dataclass


from mimo.eval.ndf.utils import util, torch_util, trimesh_util, torch3d_util
from mimo.eval.ndf.utils.plotly_save import plot3d

from mimo.reconstruction.mesh_reconstruction import MeshReconstruction
from mimo.utils.path_util import get_ndf_exp_cfg_path

cfg = load_dataclass(InferenceConfig, get_ndf_exp_cfg_path())


class OccNetOptimizer:
    def __init__(self, model, query_pts, query_pts_real_shape=None, opt_iterations=250, weighted=False,
                 noise_scale=0.0, noise_decay=0.5, single_object=False, grasp=False, shelf=False, ibs=False):
        self.recon = MeshReconstruction(model=model, device='cuda:0', cfg=cfg)
        self.grasp = grasp
        self.shelf = shelf
        self.ibs = ibs
        self.weighted = weighted
        self.model = model
        self.model_type = self.model.model_type
        self.query_pts_origin = query_pts
        if query_pts_real_shape is None:
            self.query_pts_origin_real_shape = query_pts
        else:
            self.query_pts_origin_real_shape = query_pts_real_shape

        self.loss_fn = torch.nn.L1Loss()
        if torch.cuda.is_available():
            self.dev = torch.device('cuda:0')
        else:
            self.dev = torch.device('cpu')

        if self.model is not None:
            self.model = self.model.to(self.dev)
            self.model.eval()

        self.opt_iterations = opt_iterations

        self.noise_scale = noise_scale
        self.noise_decay = noise_decay

        # if this is true, we will use the activations from the demo with the same shape as in test time
        # defaut is false, because we want to test optimizing with the set of demos that don't include
        # the test shape
        self.single_object = single_object 
        self.target_info = None
        self.demo_info = None
        self.demo_ibs = None
        self.demo_recon = None
        self.shape_completion = False
        if self.single_object:
            log_warn('\n\n**** SINGLE OBJECT SET TO TRUE, WILL *NOT* USE A NEW SHAPE AT TEST TIME, AND WILL EXPECT TARGET INFO TO BE SET****\n\n')

        self.debug_viz_path = 'debug_viz'
        self.viz_path = 'visualization'
        util.safe_makedirs(self.debug_viz_path)
        util.safe_makedirs(self.viz_path)
        self.viz_files = []

        self.rot_grid = util.generate_healpix_grid(size=1e6)
        # self.rot_grid = None
        self.recon_shape = None

    def _scene_dict(self):
        self.scene_dict = {}
        plotly_scene = {
            'xaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
            'yaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
            'zaxis': {'nticks': 16, 'range': [-0.5, 0.5]}
        }
        self.scene_dict['scene'] = plotly_scene

    def set_demo_info(self, demo_info):
        """Function to set the information for a set of multiple demonstrations

        Args:
            demo_info (list): Contains the information for the demos
        """
        self.demo_info = demo_info

    def set_demo_ibs_info(self, demo_ibs_info):
        self.demo_ibs = demo_ibs_info
        # self.ibs = True

    def set_demo_recon_info(self, demo_recon_info):
        self.demo_recon = demo_recon_info
        self.shape_completion = True

    def set_target(self, target_info):
        """
        Function to set the information about the task via the target activations
        """
        self.target_info = target_info

    def _get_query_pts_rs(self):
        # convert query points to camera frame
        query_pts_world_rs = torch.from_numpy(self.query_pts_origin_real_shape).float().to(self.dev)

        # convert query points to centered camera frame
        query_pts_world_rs_mean = query_pts_world_rs.mean(0)

        # center the query based on the it's shape mean, so that we perform optimization starting with the query at the origin
        query_pts_cam_cent_rs = query_pts_world_rs - query_pts_world_rs_mean
        query_pts_tf = np.eye(4)
        query_pts_tf[:-1, -1] = -query_pts_world_rs_mean.cpu().numpy()

        query_pts_tf_rs = query_pts_tf
        return query_pts_cam_cent_rs, query_pts_tf_rs

    def optimize_transform_implicit(self, shape_pts_world_np, ee=True, m="occ", *args, **kwargs):
        """
        Function to optimzie the transformation of our query points, conditioned on
        a set of shape points observed in the world

        Args:
            shape_pts_world (np.ndarray): N x 3 array representing 3D point cloud of the object
                to be manipulated, expressed in the world coordinate system
        """
        dev = self.dev
        n_pts = 1500
        opt_pts = 500
        perturb_scale = self.noise_scale
        perturb_decay = self.noise_decay

        if self.single_object:
            assert self.target_info is not None, 'Target info not set! Need to set the targets for single object optimization'

        # --------------------------  obtain the activations from the demos  ---------------------------------- #
        demo_feats_list = []
        demo_latents_list = []
        for i in range(len(self.demo_info)):
            if i == 10:
                break
            # load in information from target
            if self.shape_completion:
                demo_shape_pts_world = self.demo_recon[i]
            else:
                demo_shape_pts_world = self.demo_info[i]['demo_obj_pts']

            if self.demo_ibs is not None:
                if self.grasp:
                    trans_mat = util.matrix_from_pose(util.list2pose_stamped(self.demo_info[i]['demo_ee_pose_world']))
                    demo_query_pts = copy.deepcopy(self.demo_ibs)
                    ones = np.ones((demo_query_pts.shape[0], 1))
                    pts_ones = np.hstack([demo_query_pts, ones])
                    trans_ibs_pts = trans_mat @ pts_ones.T
                    demo_query_pts_world = trans_ibs_pts[:3, :].T
                else:
                    demo_query_pts_world = self.demo_ibs
            else:
                demo_query_pts_world = self.demo_info[i]['demo_query_pts']

            # higher weight for pts near the contact region
            if self.weighted:
                ref_pcd = o3d.geometry.PointCloud()
                ref_pcd.points = o3d.utility.Vector3dVector(demo_shape_pts_world)

                ref_query = o3d.geometry.PointCloud()
                ref_query.points = o3d.utility.Vector3dVector(demo_query_pts_world)

                # compute key point
                dists = ref_query.compute_point_cloud_distance(ref_pcd)
                dists = np.asarray(dists)
                weights = 1.0 / dists
                pts_weights = weights / weights.sum() * opt_pts
                self.ref_query_pts_weights = torch.from_numpy(pts_weights).float().cuda().view(1, opt_pts, 1)

            demo_shape_pts_world = torch.from_numpy(demo_shape_pts_world).float().to(self.dev)
            demo_query_pts_world = torch.from_numpy(demo_query_pts_world).float().to(self.dev)

            # centralize the pcl
            demo_shape_pts_mean = demo_shape_pts_world.mean(0)
            demo_shape_pts_cent = demo_shape_pts_world - demo_shape_pts_mean
            demo_query_pts_cent = demo_query_pts_world - demo_shape_pts_mean

            if not self.grasp and self.shelf and not self.shape_completion and self.ibs:
                demo_query_pts_cent -= torch.tensor([0, 0, 0.125], device=self.dev)
            if not self.grasp and not self.shelf and self.shape_completion and not self.ibs:  # mug
                demo_query_pts_cent -= torch.tensor([0, 0, 0.125], device=self.dev)
            # print("shelf " + str(self.shelf))
            # print("shape_completion " + str(self.shape_completion))
            # print("ibs " + str(self.ibs))

            demo_query_pts_cent_perturbed = demo_query_pts_cent + (torch.randn(demo_query_pts_cent.size()) * perturb_scale).to(dev)

            demo_shape_pts_cent_np = copy.deepcopy(demo_shape_pts_cent.cpu().numpy())
            demo_query_pts_cent_perturbed_np = copy.deepcopy(demo_query_pts_cent_perturbed.cpu().numpy())

            rndperm = torch.randperm(demo_shape_pts_cent.size(0))

            demo_model_input = dict(
                point_cloud=demo_shape_pts_cent[None, rndperm[:n_pts], :], 
                coords=demo_query_pts_cent_perturbed[None, :opt_pts, :])
            # out = self.model(demo_model_input)
            # target_act_hat = out['features'].detach()

            target_latent = self.model.extract_latent(demo_model_input).detach()
            target_act_hat = self.model.forward_latent(target_latent, demo_model_input['coords']).detach()

            if self.weighted:
                target_act_hat = target_act_hat * self.ref_query_pts_weights

            demo_feats_list.append(target_act_hat.squeeze())
            demo_latents_list.append(target_latent.squeeze())

            # # vis demo
            # ps.init()
            # ps.set_up_dir("z_up")
            #
            # ps.register_point_cloud("demo_pcd", demo_shape_pts_cent_np, radius=0.005, color=[0, 0, 1], enabled=True)
            # ps.register_point_cloud("demo_query", demo_query_pts_cent_perturbed_np, radius=0.005, color=[0, 1, 0], enabled=True)
            # ps.show()

        target_act_hat_all = torch.stack(demo_feats_list, 0)
        target_act_hat = torch.mean(target_act_hat_all, 0)

        ######################################################################

        # convert shape pts to camera frame
        shape_pts_world = torch.from_numpy(shape_pts_world_np).float().to(self.dev)
        shape_pts_mean = shape_pts_world.mean(0)
        shape_pts_cent = shape_pts_world - shape_pts_mean

        ############################################################################
        # shape = {}
        # pcd = shape_pts_cent.cpu().numpy()
        # pcd_mean = np.mean(pcd, axis=0)
        # pcd = pcd - pcd_mean
        # pcd_pts_np = copy.deepcopy(pcd)
        # np.random.shuffle(pcd)
        # shape_pcd = trimesh.PointCloud(pcd * 1.2)
        # bb = shape_pcd.bounding_box
        #
        # eval_pts = bb.sample_volume(100000)
        # shape['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(self.dev).detach()
        #
        # if not torch.is_tensor(pcd):
        #     shape['point_cloud'] = torch.from_numpy(pcd[:1000]).float().to(self.dev)[None, :, :]
        # else:
        #     shape['point_cloud'] = pcd[:1000].float().to(self.dev)[None, :, :]
        # out = self.model(shape)
        # in_inds = torch.where(out['occ'].squeeze() > 0.3)[0].cpu().numpy()
        # in_pts = eval_pts[in_inds]
        # np.random.shuffle(in_pts)
        #
        # ps.init()
        # ps.set_up_dir("z_up")
        #
        # ps.register_point_cloud("obj", pcd[:1000], radius=0.005, color=[0, 0, 1], enabled=True)
        # ps.register_point_cloud("shape", in_pts[:1000], radius=0.005, color=[0, 1, 0], enabled=True)
        #
        # ps.show()

        # ----------------------------  mimo recon -------------------------------------- #
        if self.shape_completion:
            np.random.shuffle(shape_pts_world_np)
            if shape_pts_world_np.shape[0] > 1500:
                shape_pts_world_np = shape_pts_world_np[:1500]
            shape_pts_world = torch.from_numpy(shape_pts_world_np).float().to(self.dev)
            shape_pts_mean = shape_pts_world.mean(0)
            shape_pts_cent = shape_pts_world - shape_pts_mean
            recon_mesh = self.recon.from_occupancy(pcd=shape_pts_cent.cpu().numpy())

            # recon_pts = trimesh.sample.sample_surface(recon_mesh, 2000)
            # recon_pts = recon_pts[0]

            # postprocess
            mesh = recon_mesh.as_open3d
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            cluster_area = np.asarray(cluster_area)

            largest_cluster_idx = cluster_n_triangles.argmax()
            triangles_to_remove = triangle_clusters != largest_cluster_idx
            mesh.remove_triangles_by_mask(triangles_to_remove)
            # o3d.visualization.draw_geometries([mesh_1])

            pts = mesh.sample_points_uniformly(number_of_points=2000)
            recon_pts = np.asarray(pts.points)

            # ps.init()
            # ps.set_up_dir("z_up")
            #
            # ps.register_point_cloud("recon", recon_pts, radius=0.005, color=[0, 0, 1], enabled=True)
            # ps.register_point_cloud("pts", shape_pts_cent.cpu().numpy(), radius=0.005, color=[1, 0, 0], enabled=True)
            #
            # ps.show()

            shape_pts_world = torch.from_numpy(recon_pts).float().to(self.dev) + shape_pts_mean
            self.recon_shape = shape_pts_world.cpu().numpy()
            shape_pts_mean = shape_pts_world.mean(0)
            shape_pts_cent = shape_pts_world - shape_pts_mean
        #######################################################################

        # convert query points to camera frame, and center the query based on the it's shape mean, so that we perform optimization starting with the query at the origin
        query_pts_world = torch.from_numpy(self.query_pts_origin).float().to(self.dev)
        query_pts_mean = query_pts_world.mean(0)
        query_pts_cent = query_pts_world - query_pts_mean

        query_pts_tf = np.eye(4)
        query_pts_tf[:-1, -1] = -query_pts_mean.cpu().numpy()

        if 'dgcnn' in self.model_type:
            full_opt = 5   # dgcnn can't fit 10 initialization in memory
        else:
            full_opt = 10
        best_loss = np.inf
        best_tf = np.eye(4)
        best_idx = 0
        tf_list = []
        M = full_opt

        # random initialization
        # trans = (torch.rand((M, 3)) * 0.1).float().to(dev)
        # rot = torch.rand(M, 3).float().to(dev)
        # # rot_idx = np.random.randint(self.rot_grid.shape[0], size=M)
        # # rot = torch3d_util.matrix_to_axis_angle(torch.from_numpy(self.rot_grid[rot_idx])).float()
        #
        # # rand_rot_init = (torch.rand((M, 3)) * 2*np.pi).float().to(dev)
        # rand_rot_idx = np.random.randint(self.rot_grid.shape[0], size=M)
        # rand_rot_init = torch3d_util.matrix_to_axis_angle(torch.from_numpy(self.rot_grid[rand_rot_idx])).float()
        # rand_mat_init = torch_util.angle_axis_to_rotation_matrix(rand_rot_init)
        # rand_mat_init = rand_mat_init.squeeze().float().to(dev)

        # ------------------------------------------------------- #
        # # initialize 6 poses around the object
        t1 = np.eye(3) * 0.1
        t2 = np.eye(3) * -0.1
        t3 = np.random.rand(M - 6, 3) * 0.1
        t = np.vstack([t1, t2, t3])
        trans = torch.from_numpy(t).float().to(self.dev)
        rot = torch.rand(M, 3).float().to(self.dev)

        # initialization
        r1 = np.array([[-0.3849 * np.pi, -0.3849 * np.pi, 0.3849 * np.pi],
                       [0, -0.7071 * np.pi, 0.7071 * np.pi],
                       [0, np.pi, 0],
                       [0.7698 * np.pi, -0.7698 * np.pi, 0.7698 * np.pi],
                       [-np.pi / 2, 0, 0],
                       [0.001, 0.001, 0.001]])
        r2 = np.random.rand(M - 6, 3) * 2 * np.pi
        r = np.vstack([r1, r2])

        rand_rot_init = torch.from_numpy(r).float().to(self.dev)
        rand_mat_init = torch_util.angle_axis_to_rotation_matrix(rand_rot_init)
        rand_mat_init = rand_mat_init.squeeze().float().to(self.dev)
        # ----------------------------------------------------------- #

        query_pts_cam_cent_rs, query_pts_tf_rs = self._get_query_pts_rs()
        X_rs = query_pts_cam_cent_rs[:opt_pts][None, :, :].repeat((M, 1, 1))

        # set up optimization
        X = query_pts_cent[:opt_pts][None, :, :].repeat((M, 1, 1))
        X = torch_util.transform_pcd_torch(X, rand_mat_init)
        X_rs = torch_util.transform_pcd_torch(X_rs, rand_mat_init)

        mi_point_cloud = []
        for ii in range(M):
            rndperm = torch.randperm(shape_pts_cent.size(0))
            mi_point_cloud.append(shape_pts_cent[rndperm[:n_pts]])
        mi_point_cloud = torch.stack(mi_point_cloud, 0)
        mi = dict(point_cloud=mi_point_cloud)
        shape_mean_trans = np.eye(4)
        shape_mean_trans[:-1, -1] = shape_pts_mean.cpu().numpy()
        shape_pts_world_np = shape_pts_world.cpu().numpy()

        rot.requires_grad_()
        trans.requires_grad_()
        full_opt = torch.optim.Adam([trans, rot], lr=1e-2)
        full_opt.zero_grad()

        loss_values = []

        # set up model input with shape points and the shape latent that will be used throughout
        mi['coords'] = X
        latent = self.model.extract_latent(mi).detach()

        # run optimization
        pcd_traj_list = {}
        for jj in range(M):
            pcd_traj_list[jj] = []
        for i in range(self.opt_iterations):
            T_mat = torch_util.angle_axis_to_rotation_matrix(rot).squeeze()
            noise_vec = (torch.randn(X.size()) * (perturb_scale / ((i+1)**(perturb_decay)))).to(dev)
            X_perturbed = X + noise_vec
            X_new = torch_util.transform_pcd_torch(X_perturbed, T_mat) + trans[:, None, :].repeat((1, X.size(1), 1))

            # ######################### visualize the reconstruction ##################33
            #
            # # for jj in range(M):
            # if i == 0:
            #     jj = 0
            #     shape_mi = {}
            #     shape_mi['point_cloud'] = mi['point_cloud'][jj][None, :, :].detach()
            #     shape_np = shape_mi['point_cloud'].cpu().numpy().squeeze()
            #     shape_mean = np.mean(shape_np, axis=0)
            #     inliers = np.where(np.linalg.norm(shape_np - shape_mean, 2, 1) < 0.2)[0]
            #     shape_np = shape_np[inliers]
            #     shape_pcd = trimesh.PointCloud(shape_np)
            #     bb = shape_pcd.bounding_box
            #     bb_scene = trimesh.Scene(); bb_scene.add_geometry([shape_pcd, bb])
            #
            #     eval_pts = bb.sample_volume(10000)
            #     shape_mi['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(self.dev).detach()
            #     out = self.model(shape_mi)
            #     thresh = 0.3
            #     in_inds = torch.where(out['occ'].squeeze() > thresh)[0].cpu().numpy()
            #
            #     in_pts = eval_pts[in_inds]
            #     self._scene_dict()
            #     plot3d(
            #         [in_pts, shape_np],
            #         ['blue', 'black'],
            #         osp.join(self.debug_viz_path, 'recon_overlay.html'),
            #         scene_dict=self.scene_dict,
            #         z_plane=False)

            ###############################################################################

            act_hat = self.model.forward_latent(latent, X_new)
            t_size = target_act_hat.size()

            if self.weighted:
                act_hat = act_hat * self.ref_query_pts_weights

            losses = [self.loss_fn(act_hat[ii].view(t_size), target_act_hat) for ii in range(M)]
            loss = torch.mean(torch.stack(losses))
            if i % 100 == 0:
                losses_str = ['%f' % val.item() for val in losses]
                loss_str = ', '.join(losses_str)
                log_debug(f'i: {i}, losses: {loss_str}')
            loss_values.append(loss.item())
            full_opt.zero_grad()
            loss.backward()
            full_opt.step()

        best_idx = torch.argmin(torch.stack(losses)).item()
        best_loss = losses[best_idx]
        log_debug('best loss: %f, best_idx: %d' % (best_loss, best_idx))

        # ps.init()
        # ps.set_up_dir("z_up")
        for j in range(M):
            trans_j, rot_j = trans[j], rot[j]
            transform_mat_np = torch_util.angle_axis_to_rotation_matrix(rot_j.view(1, -1)).squeeze().detach().cpu().numpy()
            transform_mat_np[:-1, -1] = trans_j.detach().cpu().numpy()

            rand_query_pts_tf = np.matmul(rand_mat_init[j].detach().cpu().numpy(), query_pts_tf)
            transform_mat_np = np.matmul(transform_mat_np, rand_query_pts_tf)
            transform_mat_np = np.matmul(shape_mean_trans, transform_mat_np)

            # ee_pts_world = util.transform_pcd(self.query_pts_origin_real_shape, transform_mat_np)
            ee_pts_world = util.transform_pcd(self.query_pts_origin, transform_mat_np)

            all_pts = [ee_pts_world, shape_pts_world_np]
            opt_fname = 'ee_pose_optimized_%d.html' % j if ee else 'rack_pose_optimized_%d.html' % j
            plot3d(
                all_pts, 
                ['black', 'purple'], 
                osp.join('visualization', opt_fname), 
                z_plane=False)
            self.viz_files.append(osp.join('visualization', opt_fname))

            if ee:
                T_mat = transform_mat_np
            else:
                T_mat = np.linalg.inv(transform_mat_np)

            if not self.grasp and self.shelf and self.shape_completion and self.ibs:
                T_mat[2, -1] -= 0.11
            if not self.grasp and self.shelf and not self.shape_completion and self.ibs:
                T_mat[2, -1] -= 0.11
            # if not self.grasp and self.shelf and self.shape_completion:
            #     T_mat[2, -1] -= 0.11
            tf_list.append(T_mat)

        #     color = [1, 0, 0]
        #     if j == best_idx:
        #         color = [0, 1, 0]
        #     ps.register_point_cloud(f"pts_{j}", ee_pts_world, radius=0.005,
        #                             color=color, enabled=True)
        #     if j == 0:
        #         ps.register_point_cloud("pcd", shape_pts_world_np, radius=0.005, color=[0, 0, 1], enabled=True)
        #
        # ps.show()

        return tf_list, best_idx
