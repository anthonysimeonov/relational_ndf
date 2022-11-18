import os, os.path as osp
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import trimesh
import meshcat, meshcat.geometry as mcg

from airobot import log_info, log_warn, log_debug, log_critical

from rndf_robot.utils import util, torch_util, trimesh_util, torch3d_util
from rndf_robot.utils.plotly_save import plot3d


class OccNetOptimizer:
    def __init__(self, model, query_pts, cfg, query_pts_real_shape=None, opt_iterations=250, 
                 noise_scale=0.0025, noise_decay=0.5, single_object=False):
        self.model = model
        self.model_type = self.model.model_type
        self.query_pts_origin = query_pts 
        if query_pts_real_shape is None:
            self.query_pts_origin_real_shape = query_pts
        else:
            self.query_pts_origin_real_shape = query_pts_real_shape

        self.loss_fn =  torch.nn.L1Loss()
        if torch.cuda.is_available():
            self.dev = torch.device('cuda:0')
        else:
            self.dev = torch.device('cpu')

        if self.model is not None:
            self.model = self.model.to(self.dev)
            self.model.eval()

        self.opt_iterations = opt_iterations
        self.cfg = cfg
        self.n_pts = self.cfg.SHAPE_PCD_PTS_N
        self.opt_pts = self.cfg.QUERY_PCD_PTS_N
        if 'dgcnn' in self.model_type:
            self.full_opt = 5   # dgcnn can't fit 10 initialization in memory
        else:
            # self.full_opt = 5
            self.full_opt = 10

        self.noise_scale = noise_scale
        self.noise_decay = noise_decay

        # if this is true, we will use the activations from the demo with the same shape as in test time
        # defaut is false, because we want to test optimizing with the set of demos that don't include
        # the test shape
        self.single_object = single_object 
        self.target_info = None
        self.demo_info = None
        if self.single_object:
            log_warn('\n\n**** SINGLE OBJECT SET TO TRUE, WILL *NOT* USE A NEW SHAPE AT TEST TIME, AND WILL EXPECT TARGET INFO TO BE SET****\n\n')

        self.debug_viz_path = 'debug_viz'
        self.viz_path = 'visualization'
        util.safe_makedirs(self.debug_viz_path)
        util.safe_makedirs(self.viz_path)
        self.viz_files =  []

        self.rot_grid = util.generate_healpix_grid(size=1e6)
        # self.rot_grid = None

        self.qp_tf = np.eye(4)
        self.setup_meshcat()

    def setup_meshcat(self, mc_vis=None):
        self.mc_vis = mc_vis

    def set_query_points(self, query_pts, query_pts_real_shape=None):
        self.query_pts_origin = query_pts 
        if query_pts_real_shape is None:
            self.query_pts_origin_real_shape = query_pts
        else:
            self.query_pts_origin_real_shape = query_pts_real_shape

    def set_query_point_tf(self, qp_tf):
        self.qp_tf = qp_tf

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

    def set_target(self, target_info):
        """
        Function to set the information about the task via the target activations
        """
        self.target_info = target_info

    def _get_query_pts_rs(self, ee=True):
        # convert query points to camera frame
        query_pts_world_rs = torch.from_numpy(self.query_pts_origin_real_shape).float().to(self.dev)

        # convert query points to centered camera frame
        query_pts_world_rs_mean = query_pts_world_rs.mean(0)

        if ee:
            # center the query based on the it's shape mean, so that we perform optimization starting with the query at the origin
            query_pts_cam_cent_rs = query_pts_world_rs
            query_pts_tf = np.eye(4)

            query_pts_tf_rs = query_pts_tf
        else:
            # center the query based on the it's shape mean, so that we perform optimization starting with the query at the origin
            query_pts_cam_cent_rs = query_pts_world_rs - query_pts_world_rs_mean
            query_pts_tf = np.eye(4)
            query_pts_tf[:-1, -1] = -query_pts_world_rs_mean.cpu().numpy()

            query_pts_tf_rs = query_pts_tf
        return query_pts_cam_cent_rs, query_pts_tf_rs

    def get_target_act_hat(self):
        assert self.demo_info is not None, 'Please call set_demo_info with information from demonstrations'
        dev = self.dev
        n_pts = self.n_pts
        opt_pts = self.opt_pts
        perturb_scale = self.noise_scale
        perturb_decay = self.noise_decay
        demo_feats_list = []
        demo_latents_list = []
        for i in range(len(self.demo_info)):
            # load in information from target
            demo_shape_pts_world = self.demo_info[i]['demo_obj_pts']
            demo_query_pts_world = self.demo_info[i]['demo_query_pts']
            demo_shape_pts_world = torch.from_numpy(demo_shape_pts_world).float().to(self.dev)
            demo_query_pts_world = torch.from_numpy(demo_query_pts_world).float().to(self.dev)

            demo_shape_pts_mean = demo_shape_pts_world.mean(0)
            demo_shape_pts_cent = demo_shape_pts_world - demo_shape_pts_mean
            demo_query_pts_cent = demo_query_pts_world - demo_shape_pts_mean
            demo_query_pts_cent_perturbed = demo_query_pts_cent + (torch.randn(demo_query_pts_cent.size()) * perturb_scale).to(dev)

            rndperm = torch.randperm(demo_shape_pts_cent.size(0))
            demo_model_input = dict(
                point_cloud=demo_shape_pts_cent[None, rndperm[:n_pts], :], 
                coords=demo_query_pts_cent_perturbed[None, :opt_pts, :])
            # out = self.model(demo_model_input)
            # target_act_hat = out['features'].detach()
            target_latent = self.model.extract_latent(demo_model_input).detach()
            target_act_hat = self.model.forward_latent(target_latent, demo_model_input['coords']).detach()

            demo_feats_list.append(target_act_hat.squeeze())
            demo_latents_list.append(target_latent.squeeze())
        target_act_hat_all = torch.stack(demo_feats_list, 0)
        target_act_hat = torch.mean(target_act_hat_all, 0)
        return target_act_hat

    def get_pose_descriptor(self, shape_pts_world_np, external_obj_pose_mat, return_shape_latent=False): 
        shape_pts_world = torch.from_numpy(shape_pts_world_np).float().to(self.dev)
        shape_pts_mean = shape_pts_world.mean(0)
        shape_pts_cent = shape_pts_world - shape_pts_mean

        rndperm = torch.randperm(shape_pts_cent.size(0))
        shape_pts_cent = shape_pts_cent[rndperm[:1500]]

        query_pts_world = util.transform_pcd(self.query_pts_origin, external_obj_pose_mat)
        query_pts_world = torch.from_numpy(query_pts_world).float().to(self.dev)
        query_pts_cent = query_pts_world - shape_pts_mean

        model_input = dict(point_cloud=shape_pts_cent.reshape(1, -1, 3), coords=query_pts_cent.reshape(1, -1, 3))

        latent = self.model.extract_latent(model_input).detach()
        descriptor = self.model.forward_latent(latent, model_input['coords']).detach()
        
        if return_shape_latent:
            return descriptor, latent
        else:
            return descriptor

    def optimize_transform_implicit(self, shape_pts_world_np, ee=True, return_score_list=False, return_final_desc=False, 
                                    target_act_hat=None, visualize=False, *args, **kwargs):
        """
        Function to optimzie the transformation of our query points, conditioned on
        a set of shape points observed in the world

        Args:
            shape_pts_world (np.ndarray): N x 3 array representing 3D point cloud of the object
                to be manipulated, expressed in the world coordinate system
            ee (bool): If True, then we are running this for the end effector. If False, we are
                running it for a placement object, and the final obtained transformation will
                by inverted
            return_score_list (bool): If True, also return the list of loss values that were obtained
                from each optimization run after all optimization iterations are complete
            return_final_desc (bool): If True, return the final descriptors obtained via optimization
            target_act_hat (torch.Tensor): Optional ability to specify a target descriptor value that
                should be used for matching in the optimization. If "None" then default method will be
                called to obtain a target descriptor value from the demonstrations
            visualize (bool): If True, show intermediate steps on meshcath
        """
        dev = self.dev
        n_pts = self.n_pts
        opt_pts = self.opt_pts
        perturb_scale = self.noise_scale
        perturb_decay = self.noise_decay

        if self.single_object:
            assert self.target_info is not None, 'Target info not set! Need to set the targets for single object optimization'

        ##### obtain the activations from the demos ####

        if target_act_hat is None:
            target_act_hat = self.get_target_act_hat()

        ######################################################################

        # convert shape pts to camera frame
        shape_pts_world = torch.from_numpy(shape_pts_world_np).float().to(self.dev)
        shape_pts_mean = shape_pts_world.mean(0)
        shape_pts_cent = shape_pts_world - shape_pts_mean

        # convert query points to camera frame, and center the query based on the it's shape mean, so that we perform optimization starting with the query at the origin
        query_pts_world = torch.from_numpy(self.query_pts_origin).float().to(self.dev)
        query_pts_mean = query_pts_world.mean(0)
        query_pts_cent = query_pts_world - query_pts_mean

        query_pts_tf = np.eye(4)
        query_pts_tf[:-1, -1] = -query_pts_mean.cpu().numpy()

        # if 'dgcnn' in self.model_type:
        #     full_opt = 5   # dgcnn can't fit 10 initialization in memory
        # else:
        #     full_opt = 10
        best_loss = np.inf
        best_tf = np.eye(4)
        best_idx = 0
        tf_list = []
        M = self.full_opt

        trans_scale = 0.2
        # trans_scale = 0.5
        trans = (torch.rand((M, 3)) * trans_scale - trans_scale/2).float().to(dev)
        # rot = torch.rand(M, 3).float().to(dev)
        rot_idx = np.random.randint(self.rot_grid.shape[0], size=M)
        rot = torch3d_util.matrix_to_axis_angle(torch.from_numpy(self.rot_grid[rot_idx])).float().to(dev)

        # rand_rot_init = (torch.rand((M, 3)) * 2*np.pi).float().to(dev)
        rand_rot_idx = np.random.randint(self.rot_grid.shape[0], size=M)
        rand_rot_init = torch3d_util.matrix_to_axis_angle(torch.from_numpy(self.rot_grid[rand_rot_idx])).float()
        rand_mat_init = torch_util.angle_axis_to_rotation_matrix(rand_rot_init)
        rand_mat_init = rand_mat_init.squeeze().float().to(dev)

        query_pts_cam_cent_rs, query_pts_tf_rs = self._get_query_pts_rs(ee=ee)
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

        if self.mc_vis is not None and visualize:
            # util.meshcat_pcd_show(self.mc_vis, shape_pts_cent.cpu().numpy(), color=[255, 0, 0], name=f'scene/opt/shape_points_centered')
            util.meshcat_pcd_show(self.mc_vis, shape_pts_cent.cpu().numpy(), color=[0, 0, 255], name=f'scene/opt/shape_points_centered')

        # run optimization
        pcd_traj_list = {}
        for jj in range(M):
            pcd_traj_list[jj] = []
        viz_i = 0
        for i in range(self.opt_iterations):
            T_mat = torch_util.angle_axis_to_rotation_matrix(rot).squeeze()
            noise_val = (perturb_scale / ((i+1)**(perturb_decay)))
            noise_vec = (torch.randn(X.size()) * noise_val - noise_val/2).to(dev)
            X_perturbed = X + noise_vec
            X_new = torch_util.transform_pcd_torch(X_perturbed, T_mat) + trans[:, None, :].repeat((1, X.size(1), 1))

            ######################### visualize the reconstruction ##################33

            # for jj in range(M):
            if i == 0:
                jj = 0
                shape_mi = {}
                shape_mi['point_cloud'] = mi['point_cloud'][jj][None, :, :].detach()
                shape_np = shape_mi['point_cloud'].cpu().numpy().squeeze()
                shape_mean = np.mean(shape_np, axis=0)
                inliers = np.where(np.linalg.norm(shape_np - shape_mean, 2, 1) < 0.2)[0]
                shape_np = shape_np[inliers]
                shape_pcd = trimesh.PointCloud(shape_np)
                bb = shape_pcd.bounding_box
                bb_scene = trimesh.Scene(); bb_scene.add_geometry([shape_pcd, bb]) 

                eval_pts = bb.sample_volume(10000)
                shape_mi['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(self.dev).detach()
                out = self.model(shape_mi)
                thresh = 0.3
                in_inds = torch.where(out['occ'].squeeze() > thresh)[0].cpu().numpy()

                in_pts = eval_pts[in_inds]
                self._scene_dict()
                plot3d(
                    [in_pts, shape_np],
                    ['blue', 'black'], 
                    osp.join(self.debug_viz_path, 'recon_overlay.html'),
                    scene_dict=self.scene_dict,
                    z_plane=False)

            ###############################################################################

            act_hat = self.model.forward_latent(latent, X_new)
            t_size = target_act_hat.size()

            losses = [self.loss_fn(act_hat[ii].view(t_size), target_act_hat) for ii in range(M)]

            loss = torch.mean(torch.stack(losses))
            if i % 100 == 0:
                losses_str = ['%f' % val.item() for val in losses]
                loss_str = ', '.join(losses_str)
                # log_info(f'i: {i}, losses: {loss_str}')
                log_debug(f'i: {i}, losses: {loss_str}')
            loss_values.append(loss.item())
            full_opt.zero_grad()
            loss.backward()
            full_opt.step()

            # visualize
            if self.mc_vis is not None and visualize:
                if i % 50 != 0:
                    continue
                else:
                    X_np = X_new.detach().cpu().numpy()
                    X_new_rs = torch_util.transform_pcd_torch(X_rs, T_mat) + trans[:, None, :].repeat((1, X_rs.size(1), 1))
                    X_rs_np = X_new_rs.detach().cpu().numpy() 
                    transform_np = T_mat.detach().cpu().numpy(); trans_np = trans.detach().cpu().numpy()
                    rand_mat_init_np = rand_mat_init.detach().cpu().numpy()
                    for ii in range(M):
                        # if ii in [0, 2, 4, 6, 8]:
                        #     continue
                        transform_ii = transform_np[ii]; transform_ii[:-1, -1] = trans_np[ii]
                        transform_ii_query = np.matmul(transform_ii, rand_mat_init_np[ii])
                        # Xii_rs = util.transform_pcd(self.query_pts_origin_real_shape, transform_ii_query)
                        Xii = X_np[ii]
                        Xii_rs = X_rs_np[ii]

                        transform_ii_full = np.matmul(transform_ii_query, self.qp_tf)

                        util.meshcat_pcd_show(self.mc_vis, Xii, color=[0, 0, 0], name=f'scene/opt/X_{ii}')
                        util.meshcat_frame_show(self.mc_vis, f'scene/opt/X_{ii}_frame', transform_ii_full, length=0.05, radius=0.002, opacity=0.9)
                        # util.meshcat_pcd_show(self.mc_vis, Xii, color=[0, 0, 0], name=f'scene/opt/X/{ii}/{viz_i}')
                        # util.meshcat_frame_show(self.mc_vis, f'scene/opt/X/{ii}_frame/{viz_i}', transform_ii_full, length=0.05, radius=0.002, opacity=0.9)
                        # util.meshcat_pcd_show(self.mc_vis, Xii_rs, color=[128, 128, 0], name=f'scene/opt/X_{ii}_rs')
                        viz_i += 1
                    # user_val = input('Press enter to continue')

        desc_losses = [losses[ii].clone().detach() for ii in range(len(losses))]
        best_idx = torch.argmin(torch.stack(losses)).item()

        best_loss = desc_losses[best_idx]
        log_debug('best loss: %f, best_idx: %d' % (best_loss, best_idx))

        for j in range(M):
            trans_j, rot_j = trans[j], rot[j]
            transform_mat_np = torch_util.angle_axis_to_rotation_matrix(rot_j.view(1, -1)).squeeze().detach().cpu().numpy()
            transform_mat_np[:-1, -1] = trans_j.detach().cpu().numpy()

            rand_query_pts_tf = np.matmul(rand_mat_init[j].detach().cpu().numpy(), query_pts_tf)
            transform_mat_np = np.matmul(transform_mat_np, rand_query_pts_tf)
            transform_mat_np = np.matmul(shape_mean_trans, transform_mat_np)

            ee_pts_world = util.transform_pcd(self.query_pts_origin_real_shape, transform_mat_np)

            all_pts = [ee_pts_world, shape_pts_world_np]
            opt_fname = 'ee_pose_optimized_%d.html' % j if ee else 'rack_pose_optimized_%d.html' % j
            plot3d(
                all_pts, 
                ['black', 'purple'], 
                osp.join('visualization', opt_fname), 
                z_plane=False)
            self.viz_files.append(osp.join('visualization', opt_fname))

            # if self.mc_vis is not None:
            #     util.meshcat_pcd_show(self.mc_vis, ee_pts_world, [0, 0, 0], name=f'scene/opt/ee_pts_world_{j}')
            #     # util.meshcat_pcd_show(self.mc_vis, shape_pts_world_np, [128, 0, 128], name=f'scene/opt/shape_pts_world_np_{j}')

            if ee:
                T_mat = transform_mat_np
            else:
                T_mat = np.linalg.inv(transform_mat_np)
            tf_list.append(T_mat)
        
        if return_score_list:
            if return_final_desc:
                return tf_list, best_idx, losses, act_hat
            else:
                return tf_list, best_idx, losses
        else:
            return tf_list, best_idx
