import os, os.path as osp
import logging
import sys
import random
import numpy as np
import time
import argparse
import copy
import signal
from multiprocessing import Process, Pipe, Queue, Manager
import psutil

import meshcat
from meshcat import geometry as mcg
from meshcat import transformations as mctf
import trimesh
import pybullet as p

from airobot.utils import common
from airobot.utils.common import euler2quat
from airobot.utils.pb_util import create_pybullet_client 

from rndf_robot.config.default_data_gen_cfg import get_data_gen_cfg_defaults
from rndf_robot.utils import util, path_util
from rndf_robot.utils.eval_gen_utils import constraint_obj_world
from rndf_robot.robot.multicam import MultiCams
from rndf_robot.utils.experiment_utils import DistractorSampler, DistractorObjManager, DistractorObjectEnv


def worker_gen(child_conn, global_dict, worker_flag_dict, seed, worker_id):
    while True:
        try:
            if not child_conn.poll(0.0001):
                continue
            msg = child_conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if msg == "INIT":
            np.random.seed(seed)
            pb_client = create_pybullet_client(gui=False, opengl_render=True, realtime=True)

            # we need to turn off file caching so memory doesn't keep growing
            p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pb_client.get_client_id())

            cfg = global_dict['cfg']

            x_low, x_high = min(cfg.OBJ_SAMPLE_X_HIGH_LOW), max(cfg.OBJ_SAMPLE_X_HIGH_LOW)
            y_low, y_high = min(cfg.OBJ_SAMPLE_Y_HIGH_LOW), max(cfg.OBJ_SAMPLE_Y_HIGH_LOW)
            z_low, z_high = min(cfg.OBJ_SAMPLE_Z_HIGH_LOW), max(cfg.OBJ_SAMPLE_Z_HIGH_LOW)
            table_z = cfg.TABLE_Z

            local_trial = 1

            proc = psutil.Process(os.getpid())

            if worker_id == 0:
                mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
                print(f'MeshCat URL: {mc_vis.url()}')
            else:
                mc_vis = None
            continue
        if msg == "RESET":
            p.resetSimulation()

            # put table at right spot
            table_ori = euler2quat([0, 0, np.pi / 2])
            table_id = pb_client.load_urdf('table/table.urdf',
                                    cfg.TABLE_POS,
                                    table_ori,
                                    scaling=cfg.TABLE_SCALING)  

            continue
        if msg == "SAMPLE":
            p.resetSimulation()

            # set up camera in the world
            cams = MultiCams(cfg.CAMERA, pb_client, n_cams=cfg.N_CAMERAS)
            cam_info = {}
            cam_info['pose_world'] = []
            for cam in cams.cams:
                cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))

            # put table at right spot
            table_ori = euler2quat([0, 0, np.pi / 2])
            table_id = pb_client.load_urdf('table/table.urdf',
                                    cfg.TABLE_POS,
                                    table_ori,
                                    scaling=cfg.TABLE_SCALING)  
            if mc_vis is not None:
                mc_vis['scene/table'].delete()
                mc_vis['scene/table'].set_object(mcg.Box([1.0, 1.0, 0.01]))
                mc_vis['scene/table'].set_transform(mctf.translation_matrix([0, 0, 0.98]))
            worker_flag_dict[worker_id] = False

            upright_orientation = common.euler2quat([0, 0, 0]).tolist()
            obj_file_to_load = global_dict['object_load_obj_file']
            save_dir = global_dict['save_dir']
            
            # sample mesh positiona and orientation on the table
            scale_high, scale_low = cfg.MESH_SCALE_HIGH, cfg.MESH_SCALE_LOW
            rand_val = lambda high, low: np.random.random() * (high - low) + low
            if args.rand_scale:
                # mesh_scale=[np.random.random() * (scale_high - scale_low) + scale_low] * 3
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
                mesh_scale=[cfg.MESH_SCALE_DEFAULT] * 3

            if args.same_pose and not args.any_pose:
                pos = [np.mean([x_high, x_low]), np.mean([y_high, y_low]), table_z]
                ori = upright_orientation
            else:
                pos = [
                        np.random.random() * (x_high - x_low) + x_low, 
                        np.random.random() * (y_high - y_low) + y_low, 
                        np.random.random() * (z_high - z_low) + z_low
                        ]
                rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi, max_theta=np.pi)
                if args.any_pose:
                    mesh = trimesh.load(obj_file_to_load)
                    stable_pose = random.sample(list(mesh.compute_stable_poses()[0]), 1)[0]
                    ori = util.pose_stamped2list(util.pose_from_matrix(stable_pose))[3:]
                    pose = util.list2pose_stamped(pos + ori)
                else:
                    pose = util.list2pose_stamped(pos + upright_orientation)
                pose_w_yaw = util.transform_pose(pose, util.pose_from_matrix(rand_yaw_T))
                pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]


            # in case we want to add extra objects that can act as occluders
            distractor_sampler = DistractorSampler(pb_client)
            distractor_objs_path = osp.join(path_util.get_rndf_obj_descriptions(), 'distractors/cuboids')
            distractor_manager = DistractorObjManager(distractor_objs_path, distractor_sampler, None, table_id)
            distractor_env = DistractorObjectEnv(cfg.DISTRACTOR, pb_client, None, distractor_manager, distractor_sampler)

            if args.occlude and np.random.random() > 0.5:
                n_occ = np.random.randint(1, args.max_occluders)
                distractor_env.sample_objects(n=n_occ)
                distractor_env.initialize_object_states(keep_away_region=pos[:-1])
                time.sleep(1.0)

            pb_client.set_step_sim(True)
            obj_id = pb_client.load_geom(
                'mesh', 
                mass=0.01, 
                mesh_scale=mesh_scale,
                visualfile=obj_file_to_load, 
                collifile=obj_file_to_load,
                base_pos=pos,
                base_ori=ori,
                rgba = [0.5, 0.2, 1, 1]) 
            
            # TODO: do we want this constraint for all classes?
            o_cid = constraint_obj_world(obj_id, pos, ori)
            p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)

            time.sleep(1.5)

            # get object pose with respect to the camera
            cam_poses = []
            cam_intrinsics = []
            depth_imgs = []
            seg_idxs = []
            obj_poses = []
            obj_pcd_pts = []
            uncropped_obj_pcd_pts = []
            table_pcd_pts = []

            obj_pose_world = p.getBasePositionAndOrientation(obj_id)
            obj_pose_world = util.list2pose_stamped(list(obj_pose_world[0]) + list(obj_pose_world[1]))
            obj_pose_world_np = util.pose_stamped2np(obj_pose_world)
            obj_velocity = p.getBaseVelocity(obj_id)
            for i, cam in enumerate(cams.cams): 
                cam_pose_world = cam_info['pose_world'][i]
                cam_poses.append(util.matrix_from_pose(cam_pose_world))
                cam_intrinsics.append(cam.cam_int_mat)
                obj_pose_camera = util.convert_reference_frame(
                    pose_source=obj_pose_world,
                    pose_frame_source=util.unit_pose(),
                    pose_frame_target=cam_pose_world
                )
                obj_pose_camera_np = util.pose_stamped2np(obj_pose_camera)

                rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
                pts_raw, _ = cam.get_pcd(in_world=True, rgb_image=rgb, depth_image=depth, depth_min=0, depth_max=np.inf)

                flat_seg = seg.flatten()
                flat_depth = depth.flatten()
                obj_inds = np.where(flat_seg == obj_id)
                table_inds = np.where(flat_seg == table_id)
                seg_depth = flat_depth[obj_inds[0]]  
                table_seg_depth = flat_depth[table_inds[0]]
                
                obj_pts = pts_raw[obj_inds[0], :]
                table_pts = pts_raw[table_inds[0], :][::int(table_inds[0].shape[0]/500)]
                obj_pcd_pts.append(util.crop_pcd(obj_pts))
                uncropped_obj_pcd_pts.append(obj_pts)
                table_pcd_pts.append(table_pts)
           
                obj_poses.append(obj_pose_camera_np)
                depth_imgs.append(seg_depth)
                seg_idxs.append(obj_inds)
            
            pix_3d = np.concatenate(obj_pcd_pts, axis=0)
            table_pix_3d = np.concatenate(table_pcd_pts, axis=0)

            if mc_vis is not None:
                util.meshcat_pcd_show(mc_vis, pix_3d, color=(255, 0, 0), name='scene/object_pcd')
                mc_vis['scene/object'].delete()
                # tmesh = trimesh.load(obj_file_to_load).apply_transform(util.matrix_from_list(obj_pose_world_np))
                # util.meshcat_trimesh_show(mc_vis, 'scene/object', tmesh)
                mc_vis['scene/object'].set_object(mcg.ObjMeshGeometry.from_file(obj_file_to_load))
                mc_vis['scene/object'].set_transform(util.scale_matrix(mesh_scale))
                mc_mat = np.matmul(util.matrix_from_pose(util.list2pose_stamped(obj_pose_world_np)), util.scale_matrix(mesh_scale))
                mc_vis['scene/object'].set_transform(mc_mat)

            if local_trial % 5 == 0:
                print('Local trial: {} for object index: {} on worker: {}\n\n'.format(local_trial, global_dict['object_loop_index'], worker_id))

            if not (np.abs(obj_velocity) > 0.01).any():
                num_samples = copy.deepcopy(global_dict['trial'])
                global_dict['trial'] += 1
                global_dict['trial_object'] += 1
                if global_dict['local_trial_start'] > 0:
                    local_trial = global_dict['local_trial_start']
                    global_dict['local_trial_start'] = 0
                local_trial += 1

                save_path = osp.join(save_dir, '{}_{}_{}.npz'.format(worker_id, local_trial, num_samples))

                obj_name = obj_file_to_load.split(path_util.get_rndf_obj_descriptions())[1][1:]
                np.savez(
                    save_path,
                    obj_file=obj_name,
                    mesh_scale=mesh_scale,
                    object_pose_cam_frame=obj_poses,
                    depth_observation=depth_imgs,
                    object_segmentation=seg_idxs,
                    point_cloud=pix_3d,
                    table_point_cloud=table_pix_3d,
                    obj_pose_world=obj_pose_world_np,
                    cam_pose_world=cam_poses,
                    cam_intrinsics=cam_intrinsics
                ) 
            else:
                print('\n\n\nobject was moving!!!\n\n\n')
                print(obj_velocity)
                time.sleep(2.0)

            pb_client.remove_body(obj_id)                         
            pb_client.remove_body(table_id)

            worker_flag_dict[worker_id] = True
            mem_usage_gb = proc.memory_info().rss / (1024.0**3)
            if mem_usage_gb > 1.8:
                logging.critical(f"\n\n\nMemory consumption too large, breaking at object {global_dict['object_loop_index']}, total samples {num_samples}, worker id {worker_id}\n\n\n")
                break
            child_conn.send('DONE')
            continue
        if msg == "END":
            break        
        time.sleep(0.001)
    print('Breaking Worker ID: ' + str(worker_id))
    child_conn.close()


class DataGenWorkerManager:
    def __init__(self, global_manager, num_workers=1):

        # thread/process for sending commands to the robot
        self.global_manager = global_manager
        self.global_dict = self.global_manager.dict()
        self.global_dict['trial'] = 0
        self.worker_flag_dict = self.global_manager.dict()        

        self.np_seed_base = 1
        self.setup_workers(num_workers)

    def setup_workers(self, num_workers):
        """Setup function to instantiate the desired number of
        workers. Pipes and Processes set up, stored internally,
        and started.
        Args:
            num_workers (int): Desired number of worker processes
        """
        worker_ids = np.arange(num_workers, dtype=np.int64).tolist()
        seeds = np.arange(self.np_seed_base, self.np_seed_base + num_workers, dtype=np.int64).tolist()

        self._worker_ids = worker_ids
        self.seeds = seeds

        self._pipes = {}
        self._processes = {}
        for i, worker_id in enumerate(self._worker_ids):
            parent, child = Pipe(duplex=True)
            self.worker_flag_dict[worker_id] = True
            proc = Process(
                target=worker_gen,
                args=(
                    child,
                    self.global_dict,
                    self.worker_flag_dict,
                    seeds[i],
                    worker_id,
                )
            )
            pipe = {}
            pipe['parent'] = parent
            pipe['child'] = child

            self._pipes[worker_id] = pipe
            self._processes[worker_id] = proc

        for i, worker_id in enumerate(self._worker_ids):
            self._processes[worker_id].start()
            self._pipes[worker_id]['parent'].send('INIT')
            print('RESET WORKER ID: ' + str(worker_id))
        print('FINISHED WORKER SETUP')

    def sample_trials(self, total_num_trials):
        num_trials = self.get_obj_trial_number()
        while num_trials < total_num_trials:
            num_trials = self.get_obj_trial_number()
            for i, worker_id in enumerate(self._worker_ids):
                if self.get_worker_ready(worker_id):
                    self._pipes[worker_id]['parent'].send('SAMPLE')
                    self.worker_flag_dict[worker_id] = False
            time.sleep(0.001)
        print('\n\n\n\nDone!\n\n\n\n')

    def get_pipes(self):
        return self._pipes

    def get_processes(self):
        return self._processes

    def get_worker_ids(self):
        return self._worker_ids

    def get_worker_ready(self, worker_id):
        return self.worker_flag_dict[worker_id]

    def get_global_info_dict(self):
        """Returns the globally shared dictionary of data
        generation information, including success rate and
        trial number

        Returns:
            dict: Dictionary of global information shared
                between workers
        """
        return self.global_dict

    def get_trial_number(self):
        return self.global_dict['trial']

    def get_obj_trial_number(self):
        return self.global_dict['trial_object']


def main(args):
    signal.signal(signal.SIGINT, util.signal_handler)

    cfg = get_data_gen_cfg_defaults()
    config_file = osp.join(path_util.get_rndf_config(), 'data_gen_cfgs', args.config_file)
    if osp.exists(config_file):
        cfg.merge_from_file(config_file)

    obj_class = args.object_class
    mesh_data_dir = osp.join(path_util.get_rndf_obj_descriptions(), args.obj_file_dir)
    assert osp.exists(mesh_data_dir), f'Directory for meshes: {mesh_data_dir} does not exist!'
    print('loading centered object model files from ' + str(mesh_data_dir))

    save_dir = osp.join(path_util.get_rndf_data(), 'training_data', args.save_dir)
    util.safe_makedirs(save_dir)

    # get train samples
    objects_raw = os.listdir(mesh_data_dir)
    objects_filtered = [fn for fn in objects_raw if '_dec' not in fn]
    total_filtered = len(objects_filtered)
    train_n = int(total_filtered * 0.9); test_n = total_filtered - train_n

    train_objects = sorted(objects_filtered)[:train_n]
    test_objects = sorted(objects_filtered)[train_n:]

    print('\n\n\nTest objects: ')
    print(test_objects)
    print('\n\n\n')
    
    # get specific number of samples to obtain
    if args.samples_per_object == 0:
        samples_per_object = args.total_samples / len(train_objects)
    else:
        samples_per_object = args.samples_per_object

    # write out these splits
    train_split_str = '\n'.join(train_objects)
    test_split_str = '\n'.join(test_objects)
    open(osp.join(path_util.get_rndf_share(), '%s_train_object_split.txt' % obj_class), 'w').write(train_split_str)
    open(osp.join(path_util.get_rndf_share(), '%s_test_object_split.txt' % obj_class), 'w').write(test_split_str)

    # set up processes
    mp_manager = Manager()
    manager = DataGenWorkerManager(mp_manager, num_workers=args.num_workers)

    # put some info in the shared dict before setting up the workers
    manager.global_dict['cfg'] = cfg
    manager.global_dict['save_dir'] = save_dir
    manager.global_dict['trial'] = 0
    manager.global_dict['local_trial_start'] = 0
    manager.global_dict['chunk_size'] = args.chunk_size
    
    manager.setup_workers(args.num_workers)


    if args.resume_i > 0:
        train_objects = train_objects[args.resume_i:]

        files_int = [int(fname.split('.npz')[0].split('_')[-1]) for fname in os.listdir(save_dir)]
        start_trial = max(files_int)
        manager.global_dict['trial'] = start_trial
        manager.global_dict['local_trial_start'] = start_trial

    if args.end_i > 0:
        stop_object_idx = args.end_i
    else:
        stop_object_idx = len(train_objects)

    # while True:
    for i, train_object in enumerate(train_objects):
        print('object: ', train_object)
        print('i: ', i + args.resume_i)

        manager.global_dict['object_loop_index'] = i + args.resume_i
        manager.global_dict['shapenet_id'] = train_object
        # manager.global_dict['object_load_obj_file'] = osp.join(shapenet_centered_models_dir, train_object + '_model_128_df.obj')
        manager.global_dict['object_load_obj_file'] = osp.join(mesh_data_dir, train_object)
        manager.global_dict['trial_object'] = 0
        manager.sample_trials(samples_per_object)

        if i >= stop_object_idx:
            print('Stopping on object index: ', i)
            break

    for i, worker_id in enumerate(manager._worker_ids):
        manager.get_pipes()[worker_id]['parent'].send('END')    

    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='debug.yaml')
    parser.add_argument('--total_samples', type=int, default=10000)
    parser.add_argument('--object_class', type=str, default='mug')
    parser.add_argument('--save_dir', type=str, default='debug_pcd_save')
    parser.add_argument('--metadata_dir', type=str, default='metadata')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--resume_i', type=int, default=0)
    parser.add_argument('--end_i', type=int, default=0)
    parser.add_argument('--chunk_size', type=int, default=100)
    parser.add_argument('--occlude', action='store_true')
    parser.add_argument('--same_pose', action='store_true')
    parser.add_argument('--rand_scale', action='store_true')
    parser.add_argument('--n_cams', type=int, default=4)
    parser.add_argument('--max_occluders', type=int, default=4)
    parser.add_argument('--samples_per_object', type=int, default=0)
    parser.add_argument('--any_pose', action='store_true')
    parser.add_argument('--obj_file_dir', type=str, required=True, help='name of directory (in object descriptions folder) where .obj files are saved')


    args = parser.parse_args()
    main(args)
