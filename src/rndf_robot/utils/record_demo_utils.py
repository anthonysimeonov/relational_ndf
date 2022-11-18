import os.path as osp
import numpy as np
import trimesh

from rndf_robot.utils import util, trimesh_util
from rndf_robot.utils.plotly_save import plot3d


def convert_wrist2tip(wrist_pose, wrist2tip_tf):
    """
    Function to convert a pose of the wrist link (nominally, panda_link8) to
    the pose of the frame in between the panda hand fingers
    """
    tip_pose = util.convert_reference_frame(
        pose_source=util.list2pose_stamped(wrist2tip_tf),
        pose_frame_target=util.unit_pose(),
        pose_frame_source=util.list2pose_stamped(wrist_pose)
    )
    return util.pose_stamped2list(tip_pose)


def convert_tip2wrist(tip_pose, tip2wrist_tf):
    """
    Function to convert a pose of the wrist link (nominally, panda_link8) to
    the pose of the frame in between the panda hand fingers
    """
    tip_pose = util.convert_reference_frame(
        pose_source=util.list2pose_stamped(tip2wrist_tf),
        pose_frame_target=util.unit_pose(),
        pose_frame_source=util.list2pose_stamped(tip_pose)
    )
    return util.pose_stamped2list(tip_pose)


class DefaultQueryPoints:
    def __init__(self, external_object_meshes=None, external_object_poses=None, 
                 external_object_cutoffs=None, external_object_scales=None,  default_origin_scale=0.035):
        # external_object_meshes is a dict. keys are names of the objects. values are meshfiles
        self.external_object_meshes = external_object_meshes
        self.external_object_poses = external_object_poses
        self.external_object_cutoffs = external_object_cutoffs
        self.external_object_scales = external_object_scales
        self.external_object_qp_dict = {}
        self.get_external_object_query_points()
        self.default_origin_scale = default_origin_scale
        self.default_origin_pts = np.random.normal(size=(500, 3)) * self.default_origin_scale
        self.custom_query_points = None
        self.custom_ee_query_pose_world = None

    def get_external_object_query_points(self):
        # obtain query points that will be used
        if self.external_object_meshes is None:
            return
        for k, v in self.external_object_meshes.items():
            qp_info = {}
            mesh_file = v
            mesh = trimesh.load_mesh(mesh_file)
            if isinstance(self.external_object_poses, dict):
                if self.external_object_poses[k] is not None:
                    mesh.apply_transform(self.external_object_poses[k])
            bb = mesh.bounding_box_oriented
            bb_extents = bb.extents
            new_bb_extents = bb.extents
            uniform_points = bb.sample_volume(2000)

            if isinstance(self.external_object_cutoffs, dict):
                if self.external_object_cutoffs[k] is not None:
                    # TODO: make this programmatic to set how we want to cutoff + scale the bounding box
                    max_z = np.max(uniform_points[:, 2])
                    min_z = np.min(uniform_points[:, 2])

                    cutoff_uniform_points = uniform_points[np.where(uniform_points[:, 2] > (max_z - (max_z - min_z)*0.3))[0]]

                    # top_inds = np.where(uniform_points[:, 2] > (max_z - (max_z - min_z)*0.6))[0]
                    # uniform_points = uniform_points[top_inds]
                    # bottom_inds = np.where(uniform_points[:, 2] < (max_z - (max_z - min_z)*0.3))[0]
                    # cutoff_uniform_points = uniform_points[bottom_inds]

                    cutoff_bb = trimesh.PointCloud(cutoff_uniform_points).bounding_box_oriented
                    uniform_points = cutoff_bb.sample_volume(2000)
            
            # scale the uniform points
            if isinstance(self.external_object_scales, dict):
                if self.external_object_scales[k] is not None:
                    uniform_points_mean = np.mean(uniform_points, axis=0)
                    uniform_points -= uniform_points_mean
                    uniform_points *= self.external_object_scales[k]
                    uniform_points += uniform_points_mean

            surface_points = mesh.sample(500)
            # trimesh_util.trimesh_show([uniform_points, surface_points])
            gaussian_points = np.random.normal(size=(500, 3))
            qp_info['mesh_file'] = mesh_file
            qp_info['mesh'] = mesh
            qp_info['bb'] = bb
            qp_info['uniform'] = uniform_points
            qp_info['surface'] = surface_points
            qp_info['gaussian'] = gaussian_points

            self.external_object_qp_dict[k] = qp_info

    def set_custom_query_points(self, pcd):
        self.custom_query_points = pcd

    def clear_custom_query_points(self):
        self.custom_query_points = None


def manually_segment_pcd(full_pcd, x, y, z, note='default', mean_inliers=False, downsample=False, show=False):

    pcd_proc_debug_str1 = f'[manually_segment_pcd] Cropping info: {note}, boundary params: '
    pcd_proc_debug_str2 = f'x: [{x[0]}, {x[1]}], y: [{y[0]}, {y[1]}], z: [{z[0]}, {z[1]}]' 
    print(pcd_proc_debug_str1 + pcd_proc_debug_str2)

    crop_pcd = util.crop_pcd(full_pcd, x=x, y=y, z=z)
    
    if mean_inliers:
        pcd_mean = np.mean(crop_pcd, axis=0)
        inliers = np.where(np.linalg.norm(crop_pcd - pcd_mean, 2, 1) < 0.2)[0]
        crop_pcd = crop_pcd[inliers]
    
    if downsample:
        perm = np.random.permutation(crop_pcd.shape[0])
        size = int(crop_pcd.shape[0])
        crop_pcd = crop_pcd[perm[:size]]

    # plot3d(
        # [crop_pcd], 
        # fname=osp.join(pcd_save_dir, save_fname), 
        # auto_scene=False,
        # scene_dict=plotly_scene_dict,
        # z_plane=False)

    if show:
        trimesh_util.trimesh_show([crop_pcd])

    return crop_pcd
