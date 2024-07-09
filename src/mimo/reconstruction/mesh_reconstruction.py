import torch
import time
import trimesh
import numpy as np

from tqdm import trange
from mimo.utils.common import make_3d_grid
from torch import distributions as dist, optim, autograd
from mimo.utils import libmcubes
from mimo.utils.libmise import MISE


class MeshReconstruction:
    def __init__(self, model, device, cfg):
        self.model = model.to(device)
        self.device = device

        self.occ = cfg.occ
        # hyper parameters
        self.padding = cfg.padding
        self.threshold = cfg.threshold
        self.threshold_logit = np.log(self.threshold) - np.log(1. - self.threshold)
        self.resolution0 = cfg.resolution0
        self.upsampling_steps = cfg.upsampling_steps
        self.refinement_steps = cfg.refinement_steps
        self.with_normals = cfg.with_normals

        self.n_pcd_pts = 1500

    def from_occupancy(self, pcd, return_stats=False):
        self.model.eval()
        stats_dict = {}

        if type(pcd) is np.ndarray:
            assert len(pcd.shape) == 2
            pcd = torch.from_numpy(pcd).float().unsqueeze(0).to(self.device)
        else:
            if pcd.shape[0] == 1:
                assert len(pcd.shape) == 3
                pcd = pcd.to(self.device)
            else:
                assert len(pcd.shape) == 2
                pcd = pcd.squeeze(0).to(self.device)

        # start algorithm
        start_time = time.time()

        pcd_np = pcd.squeeze(0).cpu().numpy()
        shape_pcd = trimesh.PointCloud(pcd_np * (1 + self.padding * 2))  # query region larger than point cloud observation
        bb = shape_pcd.bounding_box
        box_size = np.max(bb.bounds - bb.centroid) * 2.0

        n_pts = pcd.shape[1]
        # assert n_pts >= 1000
        if n_pts > 1000:
            perm_index = torch.randperm(n_pts)
            pcd = pcd[:, perm_index[:self.n_pcd_pts]]

        test_input = {"point_cloud": pcd}

        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,) * 3, (0.5,) * 3, (nx,) * 3
            )

            p = pointsf
            p_split = torch.split(p, 100000)
            predictions = []

            for pi in p_split:
                pi = pi.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    test_input["coords"] = pi

                    if self.occ:
                        logits = self.model(test_input)["occ"]
                        pred = dist.Bernoulli(logits=logits).logits
                    else:
                        sdfs = self.model(test_input)["sdf"]
                        pred = torch.where(sdfs <= 0, 1.0, 0.0)

                predictions.append(pred.squeeze(0).detach().cpu())

            values = torch.cat(predictions, dim=0).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(self.resolution0, self.upsampling_steps, self.threshold_logit)
            points = mesh_extractor.query()

            while points.shape[0] != 0:
                # Query points
                pointsf = torch.FloatTensor(points).to(self.device)
                # Normalize to bounding box
                pointsf = pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 0.5)
                # Evaluate model and update
                p = pointsf
                p_split = torch.split(p, 100000)
                predictions = []

                for pi in p_split:
                    pi = pi.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        test_input["coords"] = pi

                        if self.occ:
                            logits = self.model(test_input)["occ"]
                            pred = dist.Bernoulli(logits=logits).logits
                        else:
                            sdfs = self.model(test_input)["sdf"]
                            pred = torch.where(sdfs <= 0, 1.0, 0.0)

                    predictions.append(pred.squeeze(0).detach().cpu())

                values = torch.cat(predictions, dim=0).cpu().numpy()

                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # extract mesh
        n_x, n_y, n_z = value_grid.shape

        value_padded = np.pad(
            value_grid, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(value_padded, self.threshold_logit)

        stats_dict['time_recon'] = time.time() - start_time
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
        vertices = box_size * (vertices - 0.5)

        # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
        # mesh_pymesh = fix_pymesh(mesh_pymesh)

        normals = None

        # predict normals
        if self.with_normals and not vertices.shape[0] == 0:
            start_time = time.time()
            normals = self.estimate_normals(vertices, test_input["point_cloud"])
            stats_dict['time_normal'] = time.time() - start_time

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)

        # Refine mesh
        if self.refinement_steps > 0:
            start_time = time.time()
            self.refine_mesh(mesh, value_grid, test_input["point_cloud"])
            stats_dict['time_refine'] = time.time() - start_time

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def estimate_normals(self, vertices, pcd):
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, 100000)

        test_input = {"point_cloud": pcd}
        normals = []
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(self.device)
            vi.requires_grad_(True)

            test_input["coords"] = vi
            pred_dict = self.model(test_input)
            if self.occ:
                pred = pred_dict["occ"]
            else:
                pred = pred_dict["sdf"]

            pred.backward(torch.ones_like(pred))

            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, pred, pcd):
        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = pred.shape
        assert (n_x == n_y == n_z)

        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        # optimizer = optim.RMSprop([v], lr=1e-4)
        optimizer = optim.Adam([v], lr=1e-4)

        for it_r in trange(self.refinement_steps):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)

            test_input = {"point_cloud": pcd, "coords": face_point.unsqueeze(0)}
            pred_dict = self.model(test_input)
            if self.occ:
                face_value = pred_dict["occ"]
            else:
                face_value = pred_dict["sdf"]

            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh
