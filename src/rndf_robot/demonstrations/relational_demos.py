import os, os.path as osp
import random
import copy
import numpy as np
import time
import argparse
import threading
import signal
import trimesh
import pybullet as p
import meshcat

from airobot.utils import common
from airobot import log_info, log_warn, log_debug, log_critical, set_log_level

from rndf_robot.utils import util, path_util

from airobot import Robot

from rndf_robot.utils import util, path_util
from rndf_robot.utils.eval_gen_utils import safeCollisionFilterPair
from rndf_robot.robot.multicam import MultiCams
from rndf_robot.utils.eval_gen_utils import constraint_obj_world, safeCollisionFilterPair, safeRemoveConstraint

from rndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from rndf_robot.utils.pb2mc.pybullet_meshcat import PyBulletMeshcat

from rndf_robot.share.globals import (
    SHAPENET_ID_DICT, bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list)


MOVEMENT_KEY_MSG_MAP = {
    'a': 'Y+',
    'd': 'Y-',
    's': 'X+',
    'x': 'X-',
    'e': 'Z+',
    'q': 'Z-',
    'r': 'G+',
    'f': 'G-',
    'u': 'rX+',
    'o': 'rX-',
    'i': 'rY+',
    'k': 'rY-',
    'j': 'rZ+',
    'l': 'rZ-'
}

KEY_MSG_MAP = {
    'z': 'OPEN',
    'c': 'CLOSE',
    '0': 'END',
    '9': 'RESET',
    '1': 'DEMO_PICK',
    '2': 'DEMO_PLACE',
    '3': 'SKIP',
    '4': 'ON_RACK',
    '5': 'OFF_RACK',
    'n': 'SAMPLE',
    'v': 'END',
    'b': 'CUSTOM_QUERY',
    'g': 'SWITCH_MANIP_ID'
}

MOVEMENT_KEY_MSG_MAP_ORD = {ord(k): v for k, v in MOVEMENT_KEY_MSG_MAP.items()}
KEY_MSG_MAP_ORD = {ord(k): v for k, v in KEY_MSG_MAP.items()}


def pb2mc_update(recorder, mc_vis):
    iters = 0
    while True:
        iters += 1
        recorder.add_keyframe()
        recorder.update_meshcat_current_state(mc_vis)
        time.sleep(1/230.0)

class DefaultQueryPoints:
    def __init__(self, external_object_meshes=None, default_origin_scale=0.035):
        # external_object_meshes is a dict. keys are names of the objects. values are meshfiles
        self.external_object_meshes = external_object_meshes
        self.external_object_qp_dict = {}
        self.get_external_object_query_points()
        self.default_origin_scale = default_origin_scale
        self.default_origin_pts = np.random.normal(size=(500, 3)) * self.default_origin_scale
        self.custom_query_points = None

    def get_external_object_query_points(self):
        # obtain query points that will be used
        if self.external_object_meshes is None:
            return
        for k, v in self.external_object_meshes.items():
            qp_info = {}
            mesh_file = v
            mesh = trimesh.load_mesh(mesh_file)
            bb = mesh.bounding_box_oriented
            uniform_points = bb.sample_volume(2000)
            surface_points = mesh.sample(500)
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


def main(args):
    
    # set_log_level('debug')
    set_log_level('info')
    
    np.random.seed(args.np_seed)
    random.seed(args.np_seed)
    signal.signal(signal.SIGINT, util.signal_handler)

    #########################################################
    # Set up object assets and names

    parent_class = args.parent_class
    child_class = args.child_class
    pcl = ['parent', 'child']

    mesh_data_dirs_all = {
        'mug': 'mug_centered_obj_normalized', 
        'bottle': 'bottle_centered_obj_normalized', 
        'bowl': 'bowl_centered_obj_normalized',
        'syn_rack_easy': 'syn_racks_easy_obj_unnormalized',
        'syn_rack_hard': 'syn_racks_hard_obj_unnormalized',
        'syn_container': 'box_containers_unnormalized'
    }

    mesh_data_dirs = {}
    for k, v in mesh_data_dirs_all.items():
        if k in [parent_class, child_class]:
            mesh_data_dirs[k] = v

    mesh_data_dirs = {k: osp.join(path_util.get_rndf_obj_descriptions(), v) for k, v in mesh_data_dirs.items()}

    bad_shapenet_ids = {
        'syn_rack_easy': [],
        'syn_rack_hard': [],
        'syn_container': [],
        'bowl': bad_shapenet_bowls_ids_list,
        'mug': bad_shapenet_mug_ids_list,
        'bottle': bad_shapenet_bottles_ids_list
    }

    mesh_names = {}
    for k, v in mesh_data_dirs.items():
        # get train samples
        objects_raw = os.listdir(v)
        objects_filtered = [fn for fn in objects_raw if (fn.split('/')[-1] not in bad_shapenet_ids[k] and '_dec' not in fn)]
        # objects_filtered = objects_raw
        total_filtered = len(objects_filtered)
        train_n = int(total_filtered * 0.9); test_n = total_filtered - train_n

        train_objects = sorted(objects_filtered)[:train_n]
        test_objects = sorted(objects_filtered)[train_n:]

        log_info('\n\n\nTest objects: ')
        log_info(test_objects)
        # log_info('\n\n\n')

        mesh_names[k] = objects_filtered
    obj_classes = list(mesh_names.keys())

    save_dir = osp.join(path_util.get_rndf_data(), 'demos', 'relation_demos', args.exp)
    util.safe_makedirs(save_dir)

    #########################################################
    # Set up teleop/robot interface

    manager = util.AttrDict(global_dict={})
    manager.global_dict['checkpoint_path'] = None
    np.random.seed(args.np_seed)
    random.seed(args.np_seed)
    manager.global_dict['seed'] = args.np_seed

    # manager.global_dict['shapenet_obj_dir'] = shapenet_obj_dir
    manager.global_dict['save_dir'] = save_dir
    manager.global_dict['trial'] = 0
    manager.global_dict['config'] = args.config

    manager.global_dict['mesh_data_dirs'] = mesh_data_dirs
    manager.global_dict['mesh_names'] = mesh_names
    manager.global_dict['obj_classes'] = obj_classes

    manager.global_dict['fixed_angle'] = args.fixed_angle

    if osp.exists(osp.join(save_dir, 'demo_skipped_ids.npz')):
        skipped_ids = np.load(osp.join(save_dir, 'demo_skipped_ids.npz'))['ids'].tolist()
    else:
        skipped_ids = []

    #########################################################
    # Meshcat and robot visualization

    mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    mc_vis['scene'].delete()
    log_info(f'MeshCat URL: {mc_vis.url()}')

    global_dict = manager.global_dict

    robot = Robot('franka', pb_cfg={'gui': True, 'server': True})
    robot.pb_client.configureDebugVisualizer(robot.pb_client.COV_ENABLE_GUI, 0)
    recorder = PyBulletMeshcat(pb_client=robot.pb_client, tmp_urdf_dir=osp.join(path_util.get_rndf_obj_descriptions(), 'tmp_urdf'))
    recorder.clear()
    new_home = [
        -0.2798878477975077, 
        -0.23823885657833854, 
        0.28537688039025716, 
        -2.081827496447527, 
        0.10717202097307935, 
        1.8621456957353935, 
        0.8129974299835407                
    ]

    #########################################################
    # Configs

    cfg = get_eval_cfg_defaults()
    config_fname = osp.join(path_util.get_rndf_config(), 'eval_cfgs', global_dict['config'] + '.yaml')
    if osp.exists(config_fname):
        cfg.merge_from_file(config_fname)
    else:
        print('Config file %s does not exist, using defaults' % config_fname)
    cfg.OBJ_SAMPLE_X_HIGH_LOW = [0.3, 0.6]
    cfg.OBJ_SAMPLE_Y_HIGH_LOW = [-0.35, 0.35]
    cfg.freeze()

    p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=robot.pb_client.get_client_id())

    finger_joint_id = 9
    left_pad_id = 9
    right_pad_id = 10

    finger_force = 20

    delta = 0.01
    delta_angle = np.pi/24
    angle_N = 10

    table_urdf_fname = osp.join(path_util.get_rndf_descriptions(), 'hanging/table/table_manual.urdf')
    table_id = robot.pb_client.load_urdf(table_urdf_fname,
                            cfg.TABLE_POS, 
                            cfg.TABLE_ORI,
                            scaling=1.0)

    p.changeDynamics(table_id, -1, lateralFriction=0.2) #, linearDamping=5, angularDamping=5)

    obj_id = None
    cid = None

    x_low, x_high = cfg.OBJ_SAMPLE_X_HIGH_LOW
    y_low, y_high = cfg.OBJ_SAMPLE_Y_HIGH_LOW
    table_z = cfg.TABLE_Z

    # check if we have list of skipped ids
    if osp.exists(osp.join(save_dir, 'demo_skipped_ids.npz')):
        skipped_ids = np.load(osp.join(save_dir, 'demo_skipped_ids.npz'))['ids'].tolist() 
    else:
        skipped_ids = []

    robot.pb_client.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, False)
    robot.pb_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    robot.arm.set_jpos(new_home, ignore_physics=True)

    recorder.register_object(robot.arm.robot_id, osp.join(path_util.get_rndf_descriptions(), 'franka_panda/panda.urdf'))
    recorder.register_object(table_id, table_urdf_fname)

    rec_th = threading.Thread(target=pb2mc_update, args=(recorder, mc_vis,))
    rec_th.daemon = True
    # rec_th.start()

    ##############################

    place_iter = args.resume_iter
    while True:
        obj_id = None
        manip_obj_id = None
        static_obj_id = None

        #########################################################
        # Load objects and move robot around with keyboard

        while True:

            keys = p.getKeyboardEvents()
            msg = None
            for k, v in keys.items():
                if (k in MOVEMENT_KEY_MSG_MAP_ORD.keys() and (v & p.KEY_IS_DOWN)):
                    k_chr = chr(k)
                    msg = MOVEMENT_KEY_MSG_MAP_ORD[k]
                    break
                if (k in KEY_MSG_MAP_ORD.keys() and (v & p.KEY_WAS_TRIGGERED)):
                    k_chr = chr(k)
                    msg = KEY_MSG_MAP_ORD[k]
                    break

            if msg == "X+":
                robot.arm.move_ee_xyz([delta, 0, 0])
                continue
            if msg == "X-":
                robot.arm.move_ee_xyz([-delta, 0, 0])
                continue
            if msg == "Y+":
                robot.arm.move_ee_xyz([0, delta, 0])
                continue
            if msg == "Y-":
                robot.arm.move_ee_xyz([0, -delta, 0])
                continue
            if msg == "Z+":
                robot.arm.move_ee_xyz([0, 0, delta])
                continue
            if msg == "Z-":
                robot.arm.move_ee_xyz([0, 0, -delta])
                continue
            if msg == "rX+":
                robot.arm.rot_ee_xyz(delta_angle, 'x', N=angle_N)
                continue
            if msg == "rX-":
                robot.arm.rot_ee_xyz(-delta_angle, 'x', N=angle_N)
                continue
            if msg == "rY+":
                robot.arm.rot_ee_xyz(delta_angle, 'y', N=angle_N)
                continue
            if msg == "rY-":
                robot.arm.rot_ee_xyz(-delta_angle, 'y', N=angle_N)
                continue
            if msg == "rZ+":
                robot.arm.rot_ee_xyz(delta_angle, 'z', N=angle_N)
                continue
            if msg == "rZ-":
                robot.arm.rot_ee_xyz(-delta_angle, 'z', N=angle_N)
                continue
            if msg == "OPEN":
                p.setJointMotorControl2(robot.arm.robot_id, finger_joint_id, p.VELOCITY_CONTROL, targetVelocity=1, force=finger_force)
                p.setJointMotorControl2(robot.arm.robot_id, finger_joint_id+1, p.VELOCITY_CONTROL, targetVelocity=1, force=finger_force)

                if cid is not None:
                    p.removeConstraint(cid)
                    cid = None
                continue
            if msg == "CLOSE":
                if obj_id is not None:
                    for i in range(p.getNumJoints(robot.arm.robot_id)):
                        p.setCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=False)

                p.setJointMotorControl2(robot.arm.robot_id, finger_joint_id, p.VELOCITY_CONTROL, targetVelocity=-1, force=finger_force)
                p.setJointMotorControl2(robot.arm.robot_id, finger_joint_id+1, p.VELOCITY_CONTROL, targetVelocity=-1, force=finger_force)

                if manip_obj_id is not None:
                    obj_pose_world = p.getBasePositionAndOrientation(manip_obj_id)
                    obj_pose_world = util.list2pose_stamped(list(obj_pose_world[0]) + list(obj_pose_world[1]))

                    ee_link_id = robot.arm.ee_link_id
                    ee_pose_world = np.concatenate(robot.arm.get_ee_pose()[:2]).tolist()
                    ee_pose_world = util.list2pose_stamped(ee_pose_world)

                    obj_pose_ee = util.convert_reference_frame(
                        pose_source=obj_pose_world,
                        pose_frame_target=ee_pose_world,
                        pose_frame_source=util.unit_pose()
                    )
                    obj_pose_ee_list = util.pose_stamped2list(obj_pose_ee)

                    if cid is None:
                        cid = p.createConstraint(
                            parentBodyUniqueId=robot.arm.robot_id,
                            parentLinkIndex=ee_link_id,
                            childBodyUniqueId=manip_obj_id,
                            childLinkIndex=-1,
                            jointType=p.JOINT_FIXED,
                            jointAxis=[0, 0, 0],
                            parentFramePosition=obj_pose_ee_list[:3],
                            childFramePosition=[0, 0, 0],
                            parentFrameOrientation=obj_pose_ee_list[3:])
                continue
            if msg == 'SWITCH_MANIP_ID':
                manip_obj_id_current = copy.deepcopy(manip_obj_id)
                static_obj_id_current = copy.deepcopy(static_obj_id)
                manip_obj_id = static_obj_id_current
                static_obj_id = manip_obj_id_current
                safeRemoveConstraint(static_cid)
                for obj_class in obj_classes:
                    for obj_id in obj_ids[obj_class]:
                        if obj_id == static_obj_id:
                            obj_pose_world = np.concatenate(p.getBasePositionAndOrientation(obj_id)[:2]).tolist()
                            static_cid = constraint_obj_world(static_obj_id, obj_pose_world[:3], obj_pose_world[3:])
                continue
            if msg == "SAMPLE":

                #########################################################
                # Put new objects into the scene

                time.sleep(1.0)
                recorder.current_state_lock.acquire()
                robot.arm.reset(force_reset=True)
                recorder.current_state_lock.release()
                recorder.clear()
                robot.pb_client.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, False)
                robot.pb_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
                robot.arm.set_jpos(new_home, ignore_physics=True)

                print('Resetting robot!')
                p.changeDynamics(robot.arm.robot_id, right_pad_id, lateralFriction=1.0)
                p.changeDynamics(robot.arm.robot_id, left_pad_id, lateralFriction=1.0)

                cams = MultiCams(cfg.CAMERA, robot.pb_client, n_cams=cfg.N_CAMERAS)
                cam_info = {}
                cam_info['pose_world'] = []
                cam_info['pose_world_mat'] = []
                for cam in cams.cams:
                    cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))
                    cam_info['pose_world_mat'].append(cam.cam_ext_mat)

                # put table at right spot
                table_id = robot.pb_client.load_urdf(table_urdf_fname,
                                        cfg.TABLE_POS, 
                                        cfg.TABLE_ORI,
                                        scaling=1.0)
                p.changeDynamics(table_id, -1, lateralFriction=0.2) #, linearDamping=5, angularDamping=5)
                table_base_id = 0

                obj_files_to_load = {}
                for obj_class in obj_classes:
                    n_objs = 1
                    obj_names = random.sample(mesh_names[obj_class], n_objs)
                    if (obj_class == parent_class and args.is_parent_shapenet_obj) or (obj_class == child_class and args.is_child_shapenet_obj):
                        obj_files_to_load[obj_class] = [osp.join(mesh_data_dirs[obj_class], obj_name, 'models/model_normalized.obj') for obj_name in obj_names]
                    else:
                        obj_files_to_load[obj_class] = [osp.join(mesh_data_dirs[obj_class], obj_name) for obj_name in obj_names]

                upright_orientation_dict = {
                    'mug': common.euler2quat([np.pi/2, 0, 0]).tolist(), 
                    'bottle': common.euler2quat([np.pi/2, 0, 0]).tolist(), 
                    'bowl': common.euler2quat([np.pi/2, 0, 0]).tolist(),
                    'syn_rack_easy': common.euler2quat([0, 0, 0]).tolist(),
                    'syn_rack_hard': common.euler2quat([0, 0, 0]).tolist(),
                    'syn_container': common.euler2quat([0, 0, 0]).tolist(),
                }

                obj_ids = {}
                obj_ids_all = []
                obj_meshes_poses = []

                # INFO TO BE SAVED IN PLACING DEMO FILE
                multi_obj_id_list = []
                object_names_list = []
                obj_model_file_list = []
                obj_model_file_dec_list = []
                object_classes_list = []

                for obj_class in obj_classes:
                    obj_ids[obj_class] = []
                    obj_class_i = 0
                    for fname in obj_files_to_load[obj_class]:
                        # sample mesh positions and orientation on the table
                        if (obj_class == parent_class and args.is_parent_shapenet_obj) or (obj_class == child_class and args.is_child_shapenet_obj):
                            scale_high, scale_low = cfg.MESH_SCALE_HIGH, cfg.MESH_SCALE_LOW
                        else:
                            scale_high, scale_low = 1.1, 0.9
                        rand_val = lambda high, low: np.random.random() * (high - low) + low
                        if args.rand_mesh_scale:
                            val1 = rand_val(scale_high, scale_low)
                            val2 = rand_val(scale_high, scale_low)
                            val3 = rand_val(scale_high, scale_low)
                            sample = np.random.randint(5)
                            if sample == 0:
                                mesh_scale = [val1] * 3
                            elif sample == 1:
                                mesh_scale = [val1] * 2 + [val2] 
                            elif sample == 2:
                                mesh_scale = [val1] + [val2] * 2
                            elif sample == 3:
                                mesh_scale = [val1, val2, val1]
                            elif sample == 4:
                                mesh_scale = [val1, val2, val3]
                        else:
                            if (obj_class == parent_class and args.is_parent_shapenet_obj) or (obj_class == child_class and args.is_child_shapenet_obj):
                                mesh_scale=[cfg.MESH_SCALE_DEFAULT] * 3
                            else:
                                mesh_scale=[1.0] * 3

                        # convert mesh with vhacd
                        obj_obj_file_dec = fname.split('.obj')[0] + '_dec.obj'
                        if not osp.exists(obj_obj_file_dec):
                            print('converting via VHACD')
                            p.vhacd(
                                fname,
                                obj_obj_file_dec,
                                'log.txt',
                                concavity=0.0025,
                                alpha=0.04,
                                beta=0.05,
                                gamma=0.00125,
                                minVolumePerCH=0.0001,
                                resolution=1000000,
                                depth=20,
                                planeDownsampling=4,
                                convexhullDownsampling=4,
                                pca=0,
                                mode=0,
                                convexhullApproximation=1
                            )
                        obj_file_to_load = obj_obj_file_dec
                        mesh = trimesh.load(obj_file_to_load)
                        upright_rot = np.eye(4)
                        upright_rot[:-1, :-1] = common.quat2rot(upright_orientation_dict[obj_class])
                        mesh.apply_transform(upright_rot)
                        if args.same_pose and not args.any_pose:
                            pos = [np.mean([x_high, x_low]), np.mean([y_high, y_low]), table_z]
                            ori = upright_orientation_dict[obj_class]
                        else:
                            pos = [
                                    np.random.random() * (x_high - x_low) + x_low, 
                                    np.random.random() * (y_high - y_low) + y_low, 
                                    table_z 
                                    ]
                            rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi, max_theta=np.pi)
                            if args.any_pose:
                                rpy = np.random.rand(3) * (2 * np.pi / 3) - (np.pi / 3)
                                ori = common.euler2quat([rpy[0], rpy[1], rpy[2]]).tolist()
                                pose = util.list2pose_stamped(pos + ori)
                            else:
                                pose = util.list2pose_stamped(pos + upright_orientation_dict[obj_class])
                            pose_w_yaw = util.transform_pose(pose, util.pose_from_matrix(rand_yaw_T))
                            pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]
                        if obj_class in ['syn_rack_easy', 'syn_rack_hard']:
                            # sample a random floating pose
                            robot.pb_client.set_step_sim(True)
                            obj_id = robot.pb_client.load_geom(
                                'mesh', 
                                mass=0.01, 
                                mesh_scale=mesh_scale,
                                visualfile=obj_file_to_load, 
                                collifile=obj_file_to_load,
                                base_pos=pos,
                                base_ori=ori,
                                rgba = [0.5, 0.2, 1, 1]) 

                            o_cid = constraint_obj_world(obj_id, pos, ori)
                            robot.pb_client.set_step_sim(False)
                        else:
                            obj_id = robot.pb_client.load_geom(
                                'mesh', 
                                mass=0.01, 
                                mesh_scale=mesh_scale,
                                visualfile=obj_file_to_load, 
                                collifile=obj_file_to_load,
                                base_pos=pos,
                                base_ori=ori,
                                rgba = [0.5, 0.2, 1, 1]) 

                        for ji in range(p.getNumJoints(robot.arm.robot_id)):
                            safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=table_id, linkIndexA=ji, linkIndexB=table_base_id, enableCollision=False)

                        obj_ids[obj_class].append(obj_id)
                        obj_ids_all.append(obj_id)
                        obj_meshes_poses.append((fname, util.matrix_from_pose(pose_w_yaw), mesh_scale))
                        robot.pb_client.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
                        recorder.register_object(obj_id, fname, scaling=mesh_scale)
                        time.sleep(0.5)  # give a little delay to avoid objects colliding upon spawn

                        # add to multi object placing demo info
                        obj_model_fname = fname.split(path_util.get_rndf_descriptions())[-1].lstrip('/')
                        obj_model_dec_fname = obj_file_to_load.split(path_util.get_rndf_descriptions())[-1].lstrip('/')
                        obj_name = f'{obj_class}_{obj_class_i}'

                        if (obj_class == parent_class and args.is_parent_shapenet_obj) or (obj_class == child_class and args.is_child_shapenet_obj):
                            sid = fname.split(mesh_data_dirs[obj_class])[-1].lstrip('/').split('/')[0]
                            multi_obj_id_list.append(sid)
                        else:
                            sid = fname.split('/')[-1].replace('.obj', '')
                            multi_obj_id_list.append(sid)
                        object_names_list.append(obj_name)
                        obj_model_file_list.append(obj_model_fname)
                        obj_model_file_dec_list.append(obj_model_dec_fname)
                        object_classes_list.append(obj_class)
                        obj_class_i += 1
                
                time.sleep(1.5)
                p.changeDynamics(obj_id, -1, lateralFriction=1.0)

                # get object point cloud
                rgb_imgs = []
                depth_imgs = []
                seg_imgs = []
                seg_depth_imgs = []
                seg_obj_ids = []
                seg_idxs = []
                obj_poses = []
                obj_pcd_pts = []
                cam_poses = []
                pcd_raw = []

                start_obj_pose_world_list = []
                final_obj_pose_world_list = []
                start_obj_pointcloud_list = []
                final_obj_pointcloud_list = []

                manip_obj_id = obj_ids[child_class][0]
                static_obj_id = obj_ids[parent_class][0]
                static_cid = None

                for ji in range(p.getNumJoints(robot.arm.robot_id)):
                    safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=manip_obj_id, linkIndexA=ji, linkIndexB=-1, enableCollision=False)
                    safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=static_obj_id, linkIndexA=ji, linkIndexB=-1, enableCollision=False)
                
                i_iter = 0
                for obj_class in obj_classes:
                    for obj_id in obj_ids[obj_class]:
                        obj_pose_world = np.concatenate(p.getBasePositionAndOrientation(obj_id)[:2]).tolist()
                        print(f'Object: {object_names_list[i_iter]}, Pose: ', obj_pose_world)

                        start_obj_pose_world_list.append(obj_pose_world)
                        
                        if obj_id == static_obj_id:
                            static_cid = constraint_obj_world(static_obj_id, obj_pose_world[:3], obj_pose_world[3:])
                        i_iter += 1

                cam_intrinsics = []
                
                time.sleep(3.0)
                recorder.add_keyframe()
                recorder.update_meshcat_current_state(mc_vis)
                for i, cam in enumerate(cams.cams): 
                    cam_int = cam.cam_int_mat
                    cam_ext = cam.cam_ext_mat
                    cam_intrinsics.append(cam_int)
                    cam_poses.append(cam_ext)

                    cam_pose_world = cam_info['pose_world'][i]

                    rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
                    pts_raw, _ = cam.get_pcd(in_world=True, rgb_image=rgb, depth_image=depth, depth_min=0.0, depth_max=np.inf)
                    flat_seg = seg.flatten()
                    flat_depth = depth.flatten()

                    pcd_raw.append(pts_raw)

                    # save sample
                    depth_imgs.append(depth)
                    rgb_imgs.append(rgb)
                    seg_imgs.append(seg)

                i_iter = 0
                for obj_class in obj_classes:
                    for obj_id in obj_ids[obj_class]:

                        seg_depth_imgs_obj = []
                        seg_idxs_obj = []
                        seg_obj_ids_obj = []

                        obj_pcd_pts = []
                        for jj in range(len(depth_imgs)):
                            seg, depth = seg_imgs[jj], depth_imgs[jj]
                            flat_seg = seg.flatten()
                            flat_depth = depth.flatten()
                        
                            obj_inds = np.where(flat_seg == obj_id)[0]

                            seg_depth = flat_depth[obj_inds]  
                            seg_depth_img = np.zeros(depth.shape).flatten()
                            seg_depth_img[obj_inds] = seg_depth
                            seg_depth_img = seg_depth_img.reshape(depth.shape)

                            seg_depth_imgs_obj.append(seg_depth_img)
                            seg_idxs_obj.append(obj_inds)
                            seg_obj_ids_obj.append(obj_id)
                            
                            pts_raw = pcd_raw[jj]
                            obj_pts = pts_raw[obj_inds, :]
                            obj_pcd_pts.append(util.crop_pcd(obj_pts))

                        seg_depth_imgs.append(seg_depth_imgs_obj)
                        seg_idxs.append(seg_idxs_obj)
                        seg_obj_ids.append(obj_id)
                        
                        obj_pcd = np.concatenate(obj_pcd_pts, axis=0)
                        start_obj_pointcloud_list.append(obj_pcd)
                        name = object_names_list[i_iter]

                        recorder.meshcat_scene_lock.acquire()
                        util.meshcat_pcd_show(mc_vis, obj_pcd, [0, 0, 0], name=f'scene/obj_pcd_{name}')
                        recorder.meshcat_scene_lock.release()

                        i_iter += 1

                robot.arm.set_jpos(new_home, ignore_physics=True)
                time.sleep(1.0)
                continue
            if msg == "DEMO_PLACE":

                #########################################################
                # Save the initial, final, and relative pose of the objects, and point clouds

                # save current pose
                ee_pose_world = np.concatenate(robot.arm.get_ee_pose()[:2]).tolist()
                robot_joints = robot.arm.get_jpos()
                obj_pose_world = np.concatenate(p.getBasePositionAndOrientation(obj_id)[:2]).tolist()

                i_iter = 0
                for obj_class in obj_classes:
                    for obj_id in obj_ids[obj_class]:
                        obj_pose_world = np.concatenate(p.getBasePositionAndOrientation(obj_id)[:2]).tolist()
                        print('object pose world: ', obj_pose_world)

                        final_obj_pose_world_list.append(obj_pose_world)
                        
                        start_mat = util.matrix_from_pose(util.list2pose_stamped(start_obj_pose_world_list[i_iter]))
                        final_mat = util.matrix_from_pose(util.list2pose_stamped(obj_pose_world))
                        relative_trans = np.matmul(final_mat, np.linalg.inv(start_mat))

                        final_obj_pcd = util.transform_pcd(start_obj_pointcloud_list[i_iter], relative_trans)
                        final_obj_pointcloud_list.append(final_obj_pcd)
                        name = object_names_list[i_iter]

                        recorder.meshcat_scene_lock.acquire()
                        util.meshcat_pcd_show(mc_vis, final_obj_pcd, [255, 0, 0], name=f'scene/final_obj_pcd_{name}')
                        recorder.meshcat_scene_lock.release()

                        i_iter += 1

                place_save_path = osp.join(save_dir, f'place_demo_{place_iter}.npz')
                print(f'Saving to: {place_save_path}')
                pix, cix = object_classes_list.index(parent_class), object_classes_list.index(child_class)
                save_dict = {
                    'multi_obj_names': dict(parent=object_classes_list[pix], child=object_classes_list[cix]),
                    'multi_obj_start_pcd': dict(parent=start_obj_pointcloud_list[pix], child=start_obj_pointcloud_list[cix]),
                    'multi_obj_final_pcd': dict(parent=final_obj_pointcloud_list[pix], child=final_obj_pointcloud_list[cix]),
                    'grasp_pose_world': dict(parent=None, child=None),
                    'place_pose_world': dict(parent=None, child=None),
                    'grasp_joints': dict(parent=None, child=None),
                    'place_joints': dict(parent=None, child=None),
                    'ee_link': robot.arm.ee_link_jnt,
                    'gripper_type': None,
                    'pcd_pts': None,
                    'processed_pcd': None,
                    'rgb_imgs': rgb_imgs,
                    'depth_imgs': depth_imgs,
                    'cam_intrinsics': cam_intrinsics,
                    'cam_poses': cam_poses,
                    'multi_object_ids': dict(parent=multi_obj_id_list[pix], child=multi_obj_id_list[cix]),
                    'real_sim': 'sim', # real or sim
                    'multi_obj_start_obj_pose': dict(parent=start_obj_pose_world_list[pix], child=start_obj_pose_world_list[cix]),
                    'multi_obj_final_obj_pose': dict(parent=final_obj_pose_world_list[pix], child=final_obj_pose_world_list[cix]),
                    'multi_obj_mesh_file': dict(parent=obj_model_file_list[pix], child=obj_model_file_list[cix]),
                    'multi_obj_mesh_file_dec': dict(parent=obj_model_file_dec_list[pix], child=obj_model_file_dec_list[cix]),
		}

                np.savez(place_save_path, **save_dict)
                place_iter += 1 

                time.sleep(1.0)
                continue
            if msg == "SKIP":
                place_iter += 1
                time.sleep(1.0)
                continue
            if msg == "END":
                break
            time.sleep(0.001)
        
        for obj_class in obj_classes:
            for obj_id in obj_ids[obj_class]:
                if obj_id is not None:
                    recorder.remove_object(mc_vis, obj_id)
                    robot.pb_client.remove_body(obj_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='debug_label')
    parser.add_argument('--object_class', type=str, default='mug')
    parser.add_argument('--np_seed', type=int, default=0)
    parser.add_argument('--config', type=str, default='base_config')
    parser.add_argument('--single_instance', action='store_true')
    parser.add_argument('--fixed_angle', action='store_true')
    parser.add_argument('--rand_mesh_scale', action='store_true')
    parser.add_argument('--same_pose', action='store_true')
    parser.add_argument('--any_pose', action='store_true')
    parser.add_argument('--parent_class', type=str, required=True)
    parser.add_argument('--child_class', type=str, required=True)
    parser.add_argument('--is_parent_shapenet_obj', action='store_true')
    parser.add_argument('--is_child_shapenet_obj', action='store_true')
    parser.add_argument('--resume_iter', type=int, default=0)

    args = parser.parse_args()
    main(args)
