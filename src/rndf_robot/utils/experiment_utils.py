import os, os.path as osp
import numpy as np
import trimesh
import random
import copy
import time
import pybullet as p

from rndf_robot.utils import util


class DistractorSampler:
    def __init__(self, pb_client=None):
        self.nominal_cuboid = trimesh.creation.box([0.1, 0.1, 0.1])
        # self.nominal_cuboid.apply_scale(0.5)    
        self.max_extent = [200.0, 200.0, 200.0]
        self.min_extent = [50.0, 50.0, 50.0]

        self.length_vertex_pairs = [(0, 4), (1, 5), (3, 7), (2, 6)]
        self.width_vertex_pairs = [(1, 3), (0, 2), (5, 7), (4, 6)]
        self.height_vertex_pairs = [(1, 0), (3, 2), (5, 4), (7, 6)]

        self.length_ind = 0
        self.width_ind = 1
        self.height_ind = 2

        self.pb_client = pb_client

    def scale_axis(self, scale, vertices, axis='l'):
        assert(axis in ['l', 'w', 'h'])

        scaled_vertices = copy.deepcopy(vertices)
        if axis == 'l':
            vertex_pairs = self.length_vertex_pairs
            xyz_ind = self.length_ind
        elif axis == 'w':
            vertex_pairs = self.width_vertex_pairs
            xyz_ind = self.width_ind
        elif axis == 'h':
            vertex_pairs = self.height_vertex_pairs
            xyz_ind = self.height_ind

        current_dist = np.abs(
            vertices[vertex_pairs[0][0]][xyz_ind] - vertices[vertex_pairs[0][1]][xyz_ind])
        dist_to_add = current_dist*(scale - 1.0)/2.0

        for ind_pair in vertex_pairs:
            vert_0, vert_1 = vertices[ind_pair[0]], vertices[ind_pair[1]]

            min_vert = ind_pair[np.argmin([vert_0[xyz_ind], vert_1[xyz_ind]])]
            max_vert = ind_pair[np.argmax([vert_0[xyz_ind], vert_1[xyz_ind]])]

            scaled_vertices[min_vert][xyz_ind] -= dist_to_add
            scaled_vertices[max_vert][xyz_ind] += dist_to_add
        
        return scaled_vertices

    def clamp_vertices(self, vertices):
        for i in range(vertices.shape[0]):
            for j in range(len(self.max_extent)):
                max_extent = self.max_extent[j]/2.0
                min_extent = self.min_extent[j]/2.0

                if np.abs(vertices[i][j]) > max_extent:
                    vertices[i][j] = max_extent if vertices[i][j] > 0 else -max_extent
                if np.abs(vertices[i][j]) < min_extent:
                    vertices[i][j] = min_extent if vertices[i][j] > 0 else -min_extent
        
        return vertices

    def sample_cuboid(self, scale_list):
        """scale each axis of the cuboid by some amount
        
        Args:
            scale_list (list): [scale_l, scale_w, scale_h] values
        """
        new_cuboid = copy.deepcopy(self.nominal_cuboid)
        new_vertices = new_cuboid.vertices
        new_vertices = self.scale_axis(scale_list[0], new_vertices, 'l')
        new_vertices = self.scale_axis(scale_list[1], new_vertices, 'w')
        new_vertices = self.scale_axis(scale_list[2], new_vertices, 'h')

        # new_cuboid.vertices = self.clamp_vertices(new_vertices)
        new_cuboid.vertices = new_vertices

        return new_cuboid

    def sample_random_cuboid(self):
        scale = np.random.rand(3) * (1.5 - 0.3) + 0.3
        cuboid = self.sample_cuboid(scale.tolist())
        return cuboid

    def sample_random_cuboid_stl(self, fname):
        cuboid = trimesh.load_mesh(fname)
        return cuboid

    def sample_cuboid_pybullet(self, stl_file, pos_z=1.05, goal=False, table_id=27, scale=None):
        mesh = self.sample_random_cuboid_stl(stl_file)
        if scale is None:
            obj_scale = np.ones(3)*2.0
            obj_scale = obj_scale.tolist()
        else:
            # obj_scale = [1.025, 1.75, 1.025]
            obj_scale = scale 

        obj_id = self.pb_client.load_geom(
            shape_type='mesh', 
            visualfile=stl_file, 
            collifile=stl_file, 
            mesh_scale=obj_scale,
            base_pos=[0.45, 0, pos_z],
            rgba=[0.7, 0.2, 0.2, 1.0],
            mass=0.03)

        return obj_id, mesh

    def delete_cuboid(self, obj_id, goal_obj_id=None, keypoint_ids=None):
        if keypoint_ids is not None:
            if len(keypoint_ids) > 0:
                for kp_id in keypoint_ids:
                    self.pb_client.remove_body(kp_id)
        self.pb_client.remove_body(obj_id)
        if goal_obj_id is not None:
            self.pb_client.remove_body(goal_obj_id)
        

class DistractorObjManager(object):
    def __init__(self, cuboid_path, cuboid_sampler,
                 robot_id, table_id, fname_prefix='test_cuboid_smaller_'):
        self.sampler = cuboid_sampler
        self.cuboid_path = cuboid_path

        self.robot_id = robot_id
        self.table_id = table_id

        self.cuboid_fname_prefix = fname_prefix
        self.setup_block_set()

    def setup_block_set(self):
        self.cuboid_fnames = []
        for fname in os.listdir(self.cuboid_path):
            if fname.startswith(self.cuboid_fname_prefix):
                self.cuboid_fnames.append(osp.join(self.cuboid_path,
                                                       fname))

    def get_cuboid_fname(self):
        ind = np.random.randint(len(self.cuboid_fnames))
        return self.cuboid_fnames[ind]

    def filter_collisions(self, obj_id):
        p.setCollisionFilterPair(self.robot_id,
                                 obj_id,
                                 self.table_id,
                                 -1,
                                 enableCollision=True)

    def robot_collisions_filter(self, obj_id, enable=True):
        for jnt_id in range(self.table_id):
            p.setCollisionFilterPair(self.robot_id, obj_id, jnt_id, -1, enableCollision=enable)
        p.setCollisionFilterPair(self.robot_id,
                                    obj_id,
                                    self.table_id,
                                    -1,
                                    enableCollision=True)


class DistractorObjectEnv:
    def __init__(self, distract_cfg, pb_client, robot_id, cuboid_manager, cuboid_sampler):
        self.pb_client = pb_client
        self.robot_id = robot_id
        self.cuboid_manager = cuboid_manager
        self.cuboid_sampler = cuboid_sampler
        self.cfg = distract_cfg

        self.table_boundaries = {}
        # self.table_boundaries['x'] = np.array([0.25, 0.65])
        # self.table_boundaries['y'] = np.array([-0.55, 0.55])
        self.table_boundaries['x'] = np.asarray(self.cfg.SAMPLE_X_HIGH_LOW)
        self.table_boundaries['y'] = np.asarray(self.cfg.SAMPLE_Y_HIGH_LOW)
        self.default_z = self.cfg.SAMPLE_Z_HEIGHT 
        self.keep_away_planar_thresh = 0.05
        self.avoid_obj_id_dist_thresh = 0.01
        self._current_obj_list = []
        self.clear_current_objects()

    def object_in_boundaries(self, obj_id, x_boundaries, y_boundaries, z_boundaries=None):
        obj_pos = p.getBasePositionAndOrientation(obj_id)[0]
        in_x = obj_pos[0] >= min(x_boundaries) and obj_pos[0] <= max(x_boundaries)
        in_y = obj_pos[1] >= min(y_boundaries) and obj_pos[1] <= max(y_boundaries)
        in_z = True if z_boundaries is None else obj_pos[2] >= min(z_boundaries) and obj_pos[2] <= max(z_boundaries)
        return in_x and in_y and in_z

    def _random_table_xy(self):
        x = np.random.random() * (max(self.table_boundaries['x']) - min(self.table_boundaries['x'])) + min(self.table_boundaries['x'])
        y = np.random.random() * (max(self.table_boundaries['y']) - min(self.table_boundaries['y'])) + min(self.table_boundaries['y'])
        return x, y

    def get_random_pose_mesh(self, tmesh, keep_away_region=None):
        """Sample a random pose in the table top environment with a particular mesh object.
        This method computes a stable pose of the mesh using the internal trimesh function,
        then samples a position that is in the valid object region.

        Args:
            tmesh (Trimesh.T): [description]

        Returns:
            list: Random pose [x, y, z, qx, qy, qz, qw] in the tabletop environment
        """
        # TODO: implement functionality to be able to directly sample initial states which are not at z=0
        stable_poses = tmesh.compute_stable_poses()[0]
        if not isinstance(stable_poses, list):
            stable_poses = list(stable_poses)
        # pose = np.random.choice(stable_poses, 1)
        pose = random.sample(stable_poses, 1)[0]
        # x, y = self._random_table_xy()
        while True:
            x, y = self._random_table_xy()
            if keep_away_region is None:
                break
            else:
                x_dist, y_dist = np.abs(x - keep_away_region[0]), np.abs(y - keep_away_region[1])
                # print('x_dist, y_dist: ', x_dist, y_dist)
                if x_dist > self.keep_away_planar_thresh and y_dist > self.keep_away_planar_thresh:
                    break

        pose[0] = x
        pose[1] = y
        pose[2] = self.default_z
        return util.pose_stamped2list(util.pose_from_matrix(pose))

    def hide_show_objects(self, action='hide'):
        if len(self._current_obj_list) > 0:
            for i, obj_dict in enumerate(self._current_obj_list):
                color_data = obj_dict['color']
                if action == 'hide':
                    color = [
                        color_data[0],
                        color_data[1],
                        color_data[2],
                        0]
                    p.changeVisualShape(obj_dict['obj_id'], -1, rgbaColor=color)
                else:
                    p.changeVisualShape(obj_dict['obj_id'], -1, rgbaColor=obj_dict['color'])

    def clear_current_objects(self):
        if len(self._current_obj_list) > 0:
            for i, obj_dict in enumerate(self._current_obj_list):
                self.pb_client.remove_body(obj_dict['obj_id'])
        self._current_obj_list = []

    def get_current_obj_info(self):
        return copy.deepcopy(self._current_obj_list)

    def _sample_cuboid(self, obj_fname=None):
        if obj_fname is None:
            obj_fname = self.cuboid_manager.get_cuboid_fname()
        obj_scale = self.cfg.SCALE
        obj_id, mesh = self.cuboid_sampler.sample_cuboid_pybullet(obj_fname, scale=obj_scale)
        return mesh, obj_fname, obj_id, obj_scale

    def _disable_collisions(self, obj_id_1, obj_id_2, link_id_1=-1, link_id_2=-1):
        p.setCollisionFilterPair(
            bodyUniqueIdA=obj_id_1,
            bodyUniqueIdB=obj_id_2, 
            linkIndexA=link_id_1, 
            linkIndexB=link_id_2, 
            enableCollision=False, 
            physicsClientId=self.pb_client.get_client_id())

    def sample_objects(self, n=1, filter_collision_id=None):
        self.clear_current_objects()
        obj_ids_so_far = []
        for _ in range(n):
            mesh, cuboid_fname, obj_id, obj_scale = self._sample_cuboid()
            obj_dict = {}
            obj_dict['mesh'] = mesh
            obj_dict['fname'] = cuboid_fname
            obj_dict['obj_id'] = obj_id
            obj_dict['obj_scale'] = obj_scale 
            obj_dict['color'] = p.getVisualShapeData(obj_id)[0][7]
            if filter_collision_id is not None:
                self._disable_collisions(filter_collision_id, obj_id)
                for idx in obj_ids_so_far:
                    self._disable_collisions(idx, obj_id)
            self._current_obj_list.append(obj_dict)
            obj_ids_so_far.append(obj_id)

    def initialize_object_states(self, keep_away_region=None, avoid_obj_id=None):
        # check if current set of objects is empty
        if not len(self._current_obj_list):
            raise ValueError('Must sample objects in the environment before '
                             'initializing states')
        # save any object id that we're supposed to use for collision avoidance
        self.avoid_obj_id = avoid_obj_id
        obj_ids_so_far = []
        for i, obj_dict in enumerate(self._current_obj_list):
            while True:
                # get a random start pose and set to that pose
                start_pose = self.get_random_pose_mesh(obj_dict['mesh'], keep_away_region=keep_away_region)
                obj_id = obj_dict['obj_id']
                self.pb_client.reset_body(obj_id, start_pose[:3], start_pose[3:])
                time.sleep(0.5)

                if self.avoid_obj_id is None:
                    break
                else:
                    # check to make sure we're not colliding with the avoid object
                    closest_point_info = p.getClosestPoints(bodyA=obj_id, bodyB=avoid_obj_id, distance=self.avoid_obj_id_dist_thresh, # 0.0,
                                                            linkIndexA=-1, linkIndexB=-1,
                                                            physicsClientId=self.pb_client.get_client_id())
                    valid = len(closest_point_info) == 0
                    for obj_id_so_far in obj_ids_so_far:
                        # check to make sure we're not colliding with the objects we've sampled so far
                        closest_point_info = p.getClosestPoints(bodyA=obj_id, bodyB=obj_id_so_far, distance=0.0,
                                                                linkIndexA=-1, linkIndexB=-1,
                                                                physicsClientId=self.pb_client.get_client_id())
                        valid = valid and len(closest_point_info) == 0
                    if valid:
                        break
            obj_ids_so_far.append(obj_id)


class SegmentationAugmentation:
    def __init__(self, img_size, circle_radius_hl=[0.6, 0.05], rectangle_side_hl=[0.1, 0.025]):
        self.img_size = img_size
        s = img_size
        xx, yy = np.linspace(0, 1, s[1]), np.linspace(1, 0, s[0]) 
        gx, gy = np.meshgrid(xx, yy)
        self.pixel_pos = np.vstack([gx.ravel(), gy.ravel()]).T

        self.circle_radius_hl = circle_radius_hl
        self.rectangle_side_hl = rectangle_side_hl
        self.rand_val = lambda high, low: np.random.random() * (high - low) + low

    @staticmethod
    def pix2cont(pixel_coord, img_size):
        x, y = pixel_coord[1] / img_size[1], 1.0 - pixel_coord[0] / img_size[0]
        cont_coord = np.asarray([x, y])
        return cont_coord

    @staticmethod
    def cont2pix(cont_coord, img_size):
        x, y = cont_coord[0], cont_coord[1]
        u, v = (1.0 - y) * img_size[0], x * img_size[1]
        u, v = int(u), int(v)
        pixel_coord = np.asarray([v, u])
        return pixel_coord 

    def sample_halfspace(self, obj_mask):
        """
        Masks part of an image that is on one side of a randomly sampled halfplane

        Args:
            obj_mask (np.ndarray): Size (H, W) of boolean values
        
        Returns:
            np.ndarray: Segmentation mask size (H, W) for just the sampled halfspace
        """
        # sample a random point in the obj mask
        s = self.img_size
        obj_pix = np.where(obj_mask)

        # sample random pixel, map to normalized coords, and get random point in random direction
        center_pix = (np.random.choice(obj_pix[0]), np.random.choice(obj_pix[1]))
        center_pt = self.pix2cont(center_pix, s)
        rand_angle = np.random.random() * 2*np.pi - np.pi
        dx, dy = np.cos(rand_angle)+0.001, np.sin(rand_angle)

        # compute normal vector for line connecting two points
        slope = dy / dx
        n = np.array([-1.0*slope, 1])
        n = n / np.linalg.norm(n)
        offset = np.matmul(n, center_pt)

        line_points = np.matmul(n, self.pixel_pos.T) - offset

        if np.random.random() > 0.5:
            line_inds = np.where(line_points > 0)[0]
        else:
            line_inds = np.where(line_points < 0)[0]
        line_mask = np.zeros(s, dtype=np.uint8).reshape(-1)
        line_mask[line_inds] = 1
        line_mask = line_mask.reshape(s)
        # from rndf_robot.utils.fork_pdb import ForkablePdb; ForkablePdb().set_trace()
        return line_mask

    def sample_rectangle(self, obj_mask, outside=True):
        """
        Masks part of an image that is outside of a randomly sampled rectangle

        Args:
            obj_mask (np.ndarray): Size (H, W) of boolean values
            outside (bool): If True, we will keep points OUTSIDE of box 

        Returns:
            np.ndarray: Segmentation mask size (H, W) for everything except the rectangle
        """
        # sample a random point in the obj mask
        s = self.img_size
        obj_pix = np.where(obj_mask)

        # sample random pixel, map to normalized coords, and get random point in random direction
        center_pix = (np.random.choice(obj_pix[0]), np.random.choice(obj_pix[1]))
        center_pt = self.pix2cont(center_pix, s)

        rectangle_mask = np.ones(s, dtype=np.uint8)
        h = self.rand_val(max(self.rectangle_side_hl), min(self.rectangle_side_hl))
        w = self.rand_val(max(self.rectangle_side_hl), min(self.rectangle_side_hl))
        normals = np.array([
            [1, 0], 
            [0, 1],
            [-1, 0], 
            [0, -1],
        ])
        centers = np.array([
            [center_pt[0] + w/2.0, center_pt[1]], 
            [center_pt[0], center_pt[1] + h/2.0],
            [center_pt[0] - w/2.0, center_pt[1]],
            [center_pt[0], center_pt[1] - h/2.0]
        ])
        for i in range(4):
            n = normals[i]
            # n = np.array([-1.0*slope, 1])
            n = n / np.linalg.norm(n)
            # offset = np.matmul(n, center_pt)
            offset = np.matmul(n, centers[i])

            line_points = np.matmul(n, self.pixel_pos.T) - offset

            # line_inds = np.where(line_points > 0)[0]
            line_inds = np.where(line_points < 0)[0]

            line_mask = np.zeros(s, dtype=np.uint8).reshape(-1)
            line_mask[line_inds] = 1
            line_mask = line_mask.reshape(s)
            
            rectangle_mask = rectangle_mask & line_mask
            # print('here in rect mask')
            # from IPython import embed; embed()
        if outside:
            rectangle_mask = np.logical_not(rectangle_mask)
        return rectangle_mask

    def sample_circle(self, obj_mask, inside=True):
        """
        Masks part of an image that is on one side of a randomly sampled circle

        Args:
            obj_mask (np.ndarray): Size (H, W) of boolean values
        
        Returns:
            np.ndarray: Segmentation mask size (H, W) for just the sampled circle 
        """
        # sample a random point in the obj mask
        s = self.img_size
        obj_pix = np.where(obj_mask)

        center_pix = (np.random.choice(obj_pix[0]), np.random.choice(obj_pix[1]))
        center_pt = self.pix2cont(center_pix, s)

        r = self.rand_val(max(self.circle_radius_hl), min(self.circle_radius_hl))
        # r = self.rand_val(0.6, 0.05)
        circle_points = r**2 - (self.pixel_pos[:, 0] - center_pt[0])**2 - (self.pixel_pos[:, 1] - center_pt[1])**2

        if inside:
            circle_inds = np.where(circle_points < 0)[0]
        else:
            circle_inds = np.where(circle_points > 0)[0]

        circle_mask = np.zeros(s, dtype=np.uint8).reshape(-1)
        circle_mask[circle_inds] = 1
        circle_mask = circle_mask.reshape(s)
        # from rndf_robot.utils.fork_pdb import ForkablePdb; ForkablePdb().set_trace()
        return circle_mask
