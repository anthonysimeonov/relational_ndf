import copy

from rndf_robot.utils import util, trimesh_util


class ParentChildObjectManager:
    def __init__(self, mc_vis, parent_class_name, child_class_name, panda_hand_cls):
        self.mc_vis = mc_vis
        self.panda_hand_cls = panda_hand_cls  # either PandaHand or Robotiq2F140Hand, to be instantiated
        self._active_object = 'parent'
        self.clear()

    def clear(self):
        # point clouds
        self._parent_pcd = None
        self._child_pcd = None
        self._transformed_parent_pcd = None
        self._transformed_child_pcd = None
        
        # gripper poses and robot joints
        self._parent_grasp_pose = None
        self._parent_place_pose = None
        self._child_grasp_pose = None
        self._child_place_pose = None
        self._parent_grasp_joints = None
        self._parent_place_joints = None
        self._child_grasp_joints = None
        self._child_place_joints = None

        # visualization interfaces
        self.parent_grasp_ee_viz = self.panda_hand_cls()
        self.parent_place_ee_viz = self.panda_hand_cls()
        self.child_grasp_ee_viz = self.panda_hand_cls()
        self.child_place_ee_viz = self.panda_hand_cls()

    def get_active_object(self):
        return self._active_object

    def set_active_object(self, parent_or_child):
        """
        Sets which object we should currently be tracking. 
        Will assume that the other object stays fixed

        Args:
            parent_or_child  (str): Must be either "parent" or "child"
        """
        if parent_or_child not in ['parent', 'child']:
            log_warn('Variable "parent_or_child" must be either "parent" or "child"')
            return
        self._active_object = parent_or_child

    def flip_active_object(self):
        if self._active_object == 'parent':
            self._active_object = 'child'
        elif self._active_object == 'child':
            self._active_object = 'parent'

    def set_parent_pointcloud(self, pcd):
        self._parent_pcd = pcd

    def set_child_pointcloud(self, pcd):
        self._child_pcd = pcd

    def set_pointclouds(self, pcd_dict):
        """
        Set both parent and child point clouds at the same time

        Args:
            pcd_dict (dict): Keys are "parent" and "child". Values
                are np.ndarrays (N x 3)
        """
        self._child_pcd = pcd_dict['child']
        self._parent_pcd = pcd_dict['parent']

    def get_parent_pointcloud(self):
        return copy.deepcopy(self._parent_pcd)

    def get_child_pointcloud(self):
        return copy.deepcopy(self._child_pcd)

    def get_parent_tf_pointcloud(self):
        return copy.deepcopy(self._transformed_parent_pcd)

    def get_child_tf_pointcloud(self):
        return copy.deepcopy(self._transformed_child_pcd)

    def get_parent_grasp_pose(self):
        return copy.deepcopy(self._parent_grasp_pose)

    def get_child_grasp_pose(self):
        return copy.deepcopy(self._child_grasp_pose)

    def get_parent_place_pose(self):
        return copy.deepcopy(self._parent_place_pose)

    def get_child_place_pose(self):
        return copy.deepcopy(self._child_place_pose)

    def get_parent_grasp_joints(self):
        return copy.deepcopy(self._parent_grasp_joints)

    def get_child_grasp_joints(self):
        return copy.deepcopy(self._child_grasp_joints)

    def get_parent_place_joints(self):
        return copy.deepcopy(self._parent_place_joints)

    def get_child_place_joints(self):
        return copy.deepcopy(self._child_place_joints)

    def show_parent_pointcloud(self, color=[255, 0, 0], name='scene/parent_pcd'):
        util.meshcat_pcd_show(self.mc_vis, self._parent_pcd, color=color, name=name)

    def show_child_pointcloud(self, color=[0, 0, 255], name='scene/child_pcd'):
        util.meshcat_pcd_show(self.mc_vis, self._child_pcd, color=color, name=name) 

    def trimesh_show_parent_pointcloud(self):
        trimesh_util.trimesh_show([self._parent_pcd])

    def trimesh_show_child_pointcloud(self):
        trimesh_util.trimesh_show([self._child_pcd])

    def trimesh_show_pcds(self):
        trimesh_util.trimesh_show([self._parent_pcd, self._child_pcd])

    def apply_transform_to_current(self, tf_mat):
        if self._active_object == 'parent':
            self._transformed_parent_pcd = util.transform_pcd(self._parent_pcd, tf_mat)
        if self._active_object == 'child':
            self._transformed_child_pcd = util.transform_pcd(self._child_pcd, tf_mat)

    def set_grasp_pose(self, grasp_pose_tf):
        if self._active_object == 'parent':
            self._parent_grasp_pose = grasp_pose_tf
        if self._active_object == 'child':
            self._child_grasp_pose = grasp_pose_tf

    def set_place_pose(self, place_pose_tf):
        if self._active_object == 'parent':
            self._parent_place_pose = place_pose_tf
        if self._active_object == 'child':
            self._child_place_pose = place_pose_tf

    def visualize_current_state(self):
        # if we have objects, let's see them
        if self._parent_pcd is not None:
            util.meshcat_pcd_show(self.mc_vis, self._parent_pcd, color=[255, 0, 0], name='scene/parent_pcd')
        if self._child_pcd is not None:
            util.meshcat_pcd_show(self.mc_vis, self._child_pcd, color=[0, 0, 255], name='scene/child_pcd')
        if self._transformed_parent_pcd is not None:
            util.meshcat_pcd_show(self.mc_vis, self._transformed_parent_pcd, color=[255, 0, 128], name='scene/transformed_parent_pcd')
        if self._transformed_child_pcd is not None:
            util.meshcat_pcd_show(self.mc_vis, self._transformed_child_pcd, color=[128, 0, 255], name='scene/transformed_child_pcd')

        # if we have ee poses, let's see them
        if self._parent_grasp_pose is not None:
            # parent_grasp_mat = util.matrix_from_pose(util.list2pose_stamped(self._parent_grasp_pose))
            parent_grasp_mat = self._parent_grasp_pose
            self.parent_grasp_ee_viz.reset_pose()
            self.parent_grasp_ee_viz.transform_hand(parent_grasp_mat)
            self.parent_grasp_ee_viz.meshcat_show(self.mc_vis, name_prefix='parent_grasp_pose')

        if self._parent_place_pose is not None:
            parent_place_mat = self._parent_place_pose
            self.parent_place_ee_viz.reset_pose()
            self.parent_place_ee_viz.transform_hand(parent_place_mat)
            self.parent_place_ee_viz.meshcat_show(self.mc_vis, name_prefix='parent_place_pose')

        if self._child_grasp_pose is not None:
            child_grasp_mat = self._child_grasp_pose
            self.child_grasp_ee_viz.reset_pose()
            self.child_grasp_ee_viz.transform_hand(child_grasp_mat)
            self.child_grasp_ee_viz.meshcat_show(self.mc_vis, name_prefix='child_grasp_pose')

        if self._child_place_pose is not None:
            child_place_mat = self._child_place_pose
            self.child_place_ee_viz.reset_pose()
            self.child_place_ee_viz.transform_hand(child_place_mat)
            self.child_place_ee_viz.meshcat_show(self.mc_vis, name_prefix='child_place_pose')
        
