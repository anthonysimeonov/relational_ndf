import os, os.path as osp
import copy
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import meshcat
import meshcat.geometry as mcg

from rndf_robot.utils import util, path_util


class MeshData:
    def __init__(self, fname, pose_mat, scale=1.0, tmesh=None):
        self.fname = fname
        self.pose_mat = pose_mat
        self.scale = scale
        self.tmesh = tmesh
        self.home_pose_mat = copy.deepcopy(pose_mat)


class PandaHand:
    def __init__(self, grasp_frame=False):
        # self.hand_mesh_path = osp.join(path_util.get_rndf_descriptions(), 'franka_panda/meshes/visual/hand.obj')
        self.hand_mesh_path = osp.join(path_util.get_rndf_descriptions(), 'franka_panda/meshes/collision/hand.obj')
        self.finger_mesh_path = osp.join(path_util.get_rndf_descriptions(), 'franka_panda/meshes/visual/finger.obj') 
        self.hand = trimesh.load(self.hand_mesh_path)
        self.lf = trimesh.load(self.finger_mesh_path)
        self.rf = trimesh.load(self.finger_mesh_path)

        # offset
        offset_z = 0.0584

        l_offset = [0, 0.015, offset_z]
        r_offset = [0, -0.015, offset_z]

        # rotate
        rf_tf = np.eye(4)
        rf_tf[:-1, :-1] = R.from_euler('xyz', [0, 0, np.pi]).as_matrix()
        rf_tf[:-1, -1] = r_offset

        lf_tf = np.eye(4)
        lf_tf[:-1, -1] = l_offset

        all_tf = np.eye(4)
        if grasp_frame:
            all_tf[:-1, -1] = [0, 0, -0.105]
        else:
            # rotate
            all_tf[:-1, :-1] = R.from_euler('xyz', [0., 0., -(np.pi + np.pi/4)]).as_matrix()

        rf_tf = np.matmul(all_tf, rf_tf)
        lf_tf = np.matmul(all_tf, lf_tf)

        self.hand_home_pose = all_tf
        self.rf_home_pose = rf_tf
        self.lf_home_pose = lf_tf

        self.hand.apply_transform(all_tf)
        self.lf.apply_transform(lf_tf)
        self.rf.apply_transform(rf_tf)
        self.gripper_meshes = [self.hand, self.rf, self.lf]
        self.meshes_data_dict = {
            'hand': MeshData(self.hand_mesh_path, all_tf, tmesh=self.hand), 
            'right_finger': MeshData(self.finger_mesh_path, rf_tf, tmesh=self.rf),
            'left_finger': MeshData(self.finger_mesh_path, lf_tf, tmesh=self.lf)
            }

    def reset_pose(self):
        for k, v in self.meshes_data_dict.items():
            v.pose_mat = copy.deepcopy(v.home_pose_mat)
        self.hand = trimesh.load(self.hand_mesh_path)
        self.lf = trimesh.load(self.finger_mesh_path)
        self.rf = trimesh.load(self.finger_mesh_path)
        self.hand.apply_transform(self.hand_home_pose)
        self.lf.apply_transform(self.rf_home_pose)
        self.rf.apply_transform(self.lf_home_pose)
        self.gripper_meshes = [self.hand, self.rf, self.lf]

    def transform_mesh_list(self, mesh_list, tf):
        for mesh in mesh_list:
            mesh.apply_transform(tf)

        for k, v in self.meshes_data_dict.items():
            v.pose_mat = np.matmul(tf, v.pose_mat)

    def transform_hand(self, tf):
        self.transform_mesh_list(self.gripper_meshes, tf)

    def meshcat_show(self, mc_vis, name_prefix=''):
        for k, mesh_data in self.meshes_data_dict.items():
            mc_name = 'scene/%s%s' % (name_prefix, k)
            # mc_vis[mc_name].delete()
            mc_vis[mc_name].set_object(mcg.ObjMeshGeometry.from_file(mesh_data.fname))
            mc_vis[mc_name].set_transform(util.scale_matrix(mesh_data.scale))
            mc_mat = np.matmul(mesh_data.pose_mat, util.scale_matrix(mesh_data.scale))
            mc_vis[mc_name].set_transform(mc_mat)

class Robotiq2F140Hand:
    def __init__(self):
        # self.hand_mesh_path = osp.join(path_util.get_rndf_descriptions(), 'franka_panda/meshes/visual/hand.obj')
        self.full_hand_mesh_path = osp.join(path_util.get_rndf_descriptions(), 'franka_panda/meshes/robotiq_2f140/full_hand_2f140.obj')
        self.full_hand = trimesh.load(self.full_hand_mesh_path)

        # offset
        all_tf = np.eye(4)
        all_tf[:-1, -1] = [0, 0, -0.23]

        self.hand_home_pose = all_tf

        self.full_hand.apply_transform(all_tf)
        self.gripper_meshes = [self.full_hand]
        self.meshes_data_dict = {
            'full_hand': MeshData(self.full_hand_mesh_path, all_tf, tmesh=self.full_hand), 
            }

    def reset_pose(self):
        for k, v in self.meshes_data_dict.items():
            v.pose_mat = copy.deepcopy(v.home_pose_mat)
        self.full_hand = trimesh.load(self.full_hand_mesh_path)
        self.full_hand.apply_transform(self.hand_home_pose)
        self.gripper_meshes = [self.full_hand]

    def transform_mesh_list(self, mesh_list, tf):
        for mesh in mesh_list:
            mesh.apply_transform(tf)

        for k, v in self.meshes_data_dict.items():
            v.pose_mat = np.matmul(tf, v.pose_mat)

    def transform_hand(self, tf):
        self.transform_mesh_list(self.gripper_meshes, tf)

    def meshcat_show(self, mc_vis, name_prefix=''):
        for k, mesh_data in self.meshes_data_dict.items():
            mc_name = 'scene/%s%s' % (name_prefix, k)
            # mc_vis[mc_name].delete()
            mc_vis[mc_name].set_object(mcg.ObjMeshGeometry.from_file(mesh_data.fname))
            mc_vis[mc_name].set_transform(util.scale_matrix(mesh_data.scale))
            mc_mat = np.matmul(mesh_data.pose_mat, util.scale_matrix(mesh_data.scale))
            mc_vis[mc_name].set_transform(mc_mat)
    
