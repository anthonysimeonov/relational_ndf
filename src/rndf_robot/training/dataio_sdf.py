import os, os.path as osp
import random
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

from rndf_robot.utils import path_util, geometry


class JointShapenetSDFTrainDataset(Dataset):
    def __init__(self, obj_class='all', depth_aug=False, multiview_aug=False, phase='train'):

        # Path setup (change to folder where your training data is kept)
        training_data_root = osp.join(path_util.get_rndf_data(), 'training_data')
        mug_path = osp.join(training_data_root, 'mug_table_all_pose_4_cam_half_occ_full_rand_scale')
        bottle_path = osp.join(training_data_root, 'bottle_table_all_pose_4_cam_half_occ_full_rand_scale')
        bowl_path = osp.join(training_data_root, 'bowl_table_all_pose_4_cam_half_occ_full_rand_scale')

        sdf_root = osp.join(path_util.get_rndf_obj_descriptions(), 'sdf')
        self.shapenet_mug_sdf_dir = osp.join(sdf_root, 'mug_centered_obj_normalized_sdf')
        self.shapenet_bowl_sdf_dir = osp.join(sdf_root, 'bowl_centered_obj_normalized_sdf')
        self.shapenet_bottle_sdf_dir = osp.join(sdf_root, 'bottle_centered_obj_normalized_sdf')
        self.sdf_dirs = {'mug': self.shapenet_mug_sdf_dir, 'bowl': self.shapenet_bowl_sdf_dir, 'bottle': self.shapenet_bottle_sdf_dir}

        paths = []
        if 'mug' in obj_class:
            paths.append(mug_path)
        if 'bowl' in obj_class:
            paths.append(bowl_path)
        if 'bottle' in obj_class:
            paths.append(bottle_path)

        self.obj_class = obj_class

        print('Loading from paths: ', paths)

        files_total = []
        for path in paths:
            files = list(sorted(glob.glob(path+"/*.npz")))
            n = len(files)
            idx = int(0.9 * n)

            if phase == 'train':
                files = files[:idx]
            else:
                files = files[idx:]

            files_total.extend(files)

        self.files = files_total

        self.depth_aug = depth_aug
        self.multiview_aug = multiview_aug

        block = 128 
        bs = 1 / block
        hbs = bs * 0.5
        self.bs = bs
        self.hbs = hbs

        self.projection_mode = "perspective"

        self.cache_file = None
        self.count = 0

        print("files length ", len(self.files))

    def __len__(self):
        return len(self.files)

    def get_item(self, index):
        try:
            data = np.load(self.files[index], allow_pickle=True)
            posecam =  data['object_pose_cam_frame']  # legacy naming, used to use pose expressed in camera frame. global reference frame doesn't matter though

            idxs = list(range(posecam.shape[0]))
            random.shuffle(idxs)
            select = random.randint(1, 4)

            if self.multiview_aug:
                idxs = idxs[:select]

            poses = []
            quats = []
            for i in idxs:
                pos = posecam[i, :3]
                quat = posecam[i, 3:]

                poses.append(pos)
                quats.append(quat)

            shapenet_id = str(data['shapenet_id'].item())
            category_id = str(data['shapenet_category_id'].item())

            depths = []
            segs = []
            cam_poses = []

            for i in idxs:
                seg = data['object_segmentation'][i, 0]
                depth = data['depth_observation'][i]
                rix = np.random.permutation(depth.shape[0])[:1000]
                seg = seg[rix]
                depth = depth[rix]

                if self.depth_aug:
                    depth = depth + np.random.randn(*depth.shape) * 0.1

                segs.append(seg)
                depths.append(torch.from_numpy(depth))
                cam_poses.append(data['cam_pose_world'][i])

            # change these values depending on the intrinsic parameters of camera used to collect the data. These are what we used in pybullet
            y, x = torch.meshgrid(torch.arange(480), torch.arange(640))

            # Compute native intrinsic matrix
            sensor_half_width = 320
            sensor_half_height = 240

            vert_fov = 60 * np.pi / 180

            vert_f = sensor_half_height / np.tan(vert_fov / 2)
            hor_f = sensor_half_width / (np.tan(vert_fov / 2) * 320 / 240)

            intrinsics = np.array(
                [[hor_f, 0., sensor_half_width, 0.],
                [0., vert_f, sensor_half_height, 0.],
                [0., 0., 1., 0.]]
            )

            intrinsics = torch.from_numpy(intrinsics)

            # build depth images from data
            dp_nps = []
            for i in range(len(segs)):
                seg_mask = segs[i]
                dp_np = geometry.lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depths[i].flatten(), intrinsics[None, :, :])
                dp_np = torch.cat([dp_np, torch.ones_like(dp_np[..., :1])], dim=-1)
                dp_nps.append(dp_np)

            # load in SDF data
            if shapenet_id in os.listdir(osp.join(self.sdf_dirs[self.obj_class], 'bad_meshes')):
                return self.get_item(index=random.randint(0, self.__len__() - 1))
            if '_dec' in shapenet_id:
                shapenet_id = shapenet_id.replace('_dec', '')
            sdf_path = osp.join(self.sdf_dirs[self.obj_class], f'cloud/{shapenet_id}.npz')
            sdf_data = np.load(sdf_path)
            coord, sdf = sdf_data['coords_sdf'][:, :-1], sdf_data['coords_sdf'][:, -1]

            rix = np.random.permutation(coord.shape[0])

            coord = coord[rix[:1500]]
            label = sdf[rix[:1500]]

            offset = np.random.uniform(-self.hbs, self.hbs, coord.shape)
            coord = coord + offset
            coord = coord * data['mesh_scale']

            coord = torch.from_numpy(coord)

            # transform everything into the same frame
            transforms = []
            for quat, pos in zip(quats, poses):
                quat_list = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
                rotation_matrix = Rotation.from_quat(quat_list)
                rotation_matrix = rotation_matrix.as_matrix()

                transform = np.eye(4)
                transform[:3, :3] = rotation_matrix
                transform[:3, -1] = pos
                transform = torch.from_numpy(transform)
                transforms.append(transform)


            transform = transforms[0]
            coord = torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)
            coord = torch.sum(transform[None, :, :] * coord[:, None, :], dim=-1)
            coord = coord[..., :3]

            points_world = []

            for i, dp_np in enumerate(dp_nps):
                point_transform = torch.matmul(transform, torch.inverse(transforms[i]))
                dp_np = torch.sum(point_transform[None, :, :] * dp_np[:, None, :], dim=-1)
                points_world.append(dp_np[..., :3])

            point_cloud = torch.cat(points_world, dim=0)

            rix = torch.randperm(point_cloud.size(0))
            point_cloud = point_cloud[rix[:1000]]

            if point_cloud.size(0) != 1000:
                return self.get_item(index=random.randint(0, self.__len__() - 1))

            # translate everything to the origin based on the point cloud mean
            center = point_cloud.mean(dim=0)
            coord = coord - center[None, :]
            point_cloud = point_cloud - center[None, :]

            labels = label

            # at the end we have 3D point cloud observation from depth images, voxel occupancy values and corresponding voxel coordinates
            res = {'point_cloud': point_cloud.float(),
                   'coords': coord.float(),
                   'intrinsics': intrinsics.float(),
                   'cam_poses': np.zeros(1)}  # cam poses not used
            return res, {'sdf': torch.from_numpy(labels).float()}

        except Exception as e:
           print(e)
           return self.get_item(index=random.randint(0, self.__len__() - 1))


    def __getitem__(self, index):
        return self.get_item(index)


class SynObjSDFDataset(Dataset):
    def __init__(self, obj_class='syn_rack_easy',
                 single_view=False, depth_aug=False, 
                 multiview_aug=False, phase='train'):

        # Path setup
        assert obj_class in ['syn_rack_easy', 'syn_rack_med', 'syn_container'], 'Unrecognized object class'
        training_data_root = osp.join(path_util.get_rndf_data(), 'training_data')
        syn_rack_easy_path = osp.join(training_data_root, 'syn_rack_rand_scale_easy')
        syn_rack_med_path = osp.join(training_data_root, 'syn_rack_med_pcd_rand_scale')
        syn_container_path = osp.join(training_data_root, 'syn_container_pcd_smaller')

        sdf_root = osp.join(path_util.get_rndf_obj_descriptions(), 'sdf')
        syn_rack_easy_sdf = osp.join(sdf_root, 'syn_racks_easy_obj_norm_factor_sdf')
        syn_rack_med_sdf = osp.join(sdf_root, 'syn_rack_med_norm_factor_sdf')
        syn_container_sdf = osp.join(sdf_root, 'box_containers_sdf')

        if obj_class == 'syn_rack_easy':
            data_path = syn_rack_easy_path 
            self.sdf_path = syn_rack_easy_sdf
        elif obj_class == 'syn_rack_med':
            data_path = syn_rack_med_path 
            self.sdf_path = syn_rack_med_sdf
        elif obj_class == 'syn_container':
            data_path = syn_container_path 
            self.sdf_path = syn_container_sdf
        else:
            raise ValueError('Unrecognized object class')

        files = list(sorted(glob.glob(data_path+"/*.npz")))
        n = len(files)
        idx = int(0.9 * n)

        if phase == 'train':
            files = files[:idx]
        else:
            files = files[idx:]

        self.files = files

        block = 128 
        bs = 1 / block
        hbs = bs * 0.5
        self.hbs = hbs

        self.depth_aug = depth_aug
        self.multiview_aug = multiview_aug
        self.single_view = single_view

        self.cache_file = None
        self.count = 0

        print("files length ", len(self.files))

        plotly_scene = {
            'xaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
            'yaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
            'zaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
        }

        self.scene_dict = {}
        self.scene_dict['scene'] = plotly_scene

    def __len__(self):
        return len(self.files)

    def get_item(self, index):
        data = np.load(self.files[index], allow_pickle=True)
        posecam =  data['object_pose_cam_frame']

        idxs = list(range(posecam.shape[0]))
        random.shuffle(idxs)

        select = random.randint(0, 3)
        if self.single_view:
            idxs = [idxs[select]]

        poses = []
        quats = []
        for i in idxs:
            pos = posecam[i, :3]
            quat = posecam[i, 3:]

            poses.append(pos)
            quats.append(quat)

        depths = []
        segs = []
        cam_poses = []

        for i in idxs:
            seg = data['object_segmentation'][i, 0]
            depth = data['depth_observation'][i]
            rix = np.random.permutation(depth.shape[0])[:1000]
            seg = seg[rix]
            depth = depth[rix]

            if self.depth_aug:
                depth = depth + np.random.randn(*depth.shape) * 0.1

            segs.append(seg)
            depths.append(torch.from_numpy(depth))
            cam_poses.append(data['cam_pose_world'][i])

        y, x = torch.meshgrid(torch.arange(480), torch.arange(640))

        # Compute native intrinsic matrix
        sensor_half_width = 320
        sensor_half_height = 240

        vert_fov = 60 * np.pi / 180

        vert_f = sensor_half_height / np.tan(vert_fov / 2)
        hor_f = sensor_half_width / (np.tan(vert_fov / 2) * 320 / 240)

        intrinsics = np.array(
            [[hor_f, 0., sensor_half_width, 0.],
             [0., vert_f, sensor_half_height, 0.],
             [0., 0., 1., 0.]]
        )

        intrinsics = torch.from_numpy(intrinsics)

        dp_nps = []

        for i in range(len(segs)):
            seg_mask = segs[i]
            dp_np = geometry.lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depths[i].flatten(), intrinsics[None, :, :])
            dp_np = torch.cat([dp_np, torch.ones_like(dp_np[..., :1])], dim=-1)
            dp_nps.append(dp_np)

        obj_file = data['obj_file'].item().split('/')[-1].replace('.obj', '')

        if obj_file in os.listdir(osp.join(self.sdf_path, 'bad_meshes')):
            return self.get_item(index=random.randint(0, self.__len__() - 1))
        if '_dec' in obj_file:
            obj_file = obj_file.replace('_dec', '')
        sdf_path = osp.join(self.sdf_path, f'cloud/{obj_file}.npz')
        sdf_data = np.load(sdf_path)

        coord, sdf = sdf_data['coords_sdf'][:, :-1], sdf_data['coords_sdf'][:, -1]
        norm_factor = sdf_data['norm_factor'].item()
        # print('norm factor: ', norm_factor)

        rix = np.random.permutation(coord.shape[0])

        coord = coord[rix[:1500]]
        label = sdf[rix[:1500]]

        offset = np.random.uniform(-self.hbs, self.hbs, coord.shape)
        coord = coord + offset
        coord = coord * norm_factor
        coord = coord * data['mesh_scale']

        coord = torch.from_numpy(coord)

        transforms = []
        for quat, pos in zip(quats, poses):
            quat_list = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
            rotation_matrix = Rotation.from_quat(quat_list)
            rotation_matrix = rotation_matrix.as_matrix()

            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, -1] = pos
            transform = torch.from_numpy(transform)
            transforms.append(transform)


        transform = transforms[0]
        coord = torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)
        coord = torch.sum(transform[None, :, :] * coord[:, None, :], dim=-1)
        coord = coord[..., :3]

        dp_np_extra = []

        for i, dp_np in enumerate(dp_nps):
            point_transform = torch.matmul(transform, torch.inverse(transforms[i]))
            dp_np = torch.sum(point_transform[None, :, :] * dp_np[:, None, :], dim=-1)
            dp_np_extra.append(dp_np[..., :3])

        point_cloud = torch.cat(dp_np_extra, dim=0)

        rix = torch.randperm(point_cloud.size(0))
        point_cloud = point_cloud[rix[:1000]]

        if point_cloud.size(0) != 1000:
            return self.get_item(index=random.randint(0, self.__len__() - 1))

        center = point_cloud.mean(dim=0)
        coord = coord - center[None, :]
        point_cloud = point_cloud - center[None, :]

        res = {'point_cloud': point_cloud.float(),
               'coords': coord.float(),
               'intrinsics':intrinsics.float(),
               'cam_poses': np.zeros(1)}
        return res, {'sdf': torch.from_numpy(label).float()}

    def __getitem__(self, index):
        return self.get_item(index)
