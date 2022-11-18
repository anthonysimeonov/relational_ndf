import os.path as osp
import argparse
from multiprocessing import Pool
import numpy as np
import trimesh
import trimesh.creation as cr
import meshcat

from airobot.utils import common

from rndf_robot.utils import util, path_util
from rndf_robot.utils.mesh_util import inside_mesh
from rndf_robot.utils.mesh_util.three_util import get_raster_points

from syn_rack_cfg import get_syn_rack_default_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--n_pts', type=int, default=100000)
parser.add_argument('--voxel_resolution', type=int, default=128)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--occ_save_dir', type=str, required=True)
parser.add_argument('--mesh_save_dir', type=str, required=True)
parser.add_argument('--obj_name', type=str, required=True)
parser.add_argument('--n_objs', type=int, default=200)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--only_occ', action='store_true')
parser.add_argument('--save_occ', action='store_true')
parser.add_argument('--only_center', action='store_true')
parser.add_argument('--difficulty', type=str, default='easy')

args = parser.parse_args()

cfg = get_syn_rack_default_cfg()
if args.difficulty == 'easy':
    rack_cfg = cfg.EASY
elif args.difficulty == 'med':
    rack_cfg = cfg.MED
else:
    rack_cfg = cfg.HARD

mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')

# path that we will save synthesized meshes in
mesh_save_dir = osp.join(path_util.get_rndf_obj_descriptions(), args.mesh_save_dir)
unnormalized_mesh_save_dir = osp.join(path_util.get_rndf_obj_descriptions(), args.mesh_save_dir + '_unnormalized')

# path that  we will save occupancy data in
occ_save_dir = osp.join(path_util.get_rndf_obj_descriptions(), args.occ_save_dir)

util.safe_makedirs(mesh_save_dir)
util.safe_makedirs(unnormalized_mesh_save_dir)
util.safe_makedirs(occ_save_dir)


def sample_peg(peg_cyl_radius, peg_cyl_height, base_cyl_radius, base_cyl_height, rack_cfg, top_bottom_half=None):
    peg_cyl = cr.cylinder(peg_cyl_radius, peg_cyl_height)

    # create peg pose
    half_min = 0.5
    if top_bottom_half is None:
        peg_base_height_frac = util.rand_high_low(rack_cfg.PEG_BASE_HEIGHT_FRAC_LOW_HIGH)
    elif top_bottom_half == 'top':
        peg_base_height_frac = util.rand_high_low((max(rack_cfg.PEG_BASE_HEIGHT_FRAC_LOW_HIGH), half_min))
    elif top_bottom_half == 'bottom':
        peg_base_height_frac = util.rand_high_low((half_min, min(rack_cfg.PEG_BASE_HEIGHT_FRAC_LOW_HIGH)))
    else:
        peg_base_height_frac = util.rand_high_low(rack_cfg.PEG_BASE_HEIGHT_FRAC_LOW_HIGH)
    peg_base_height = peg_base_height_frac * base_cyl_height
    peg_angle = util.rand_high_low(np.deg2rad(rack_cfg.PEG_ANGLE_LOW_HIGH))
    peg_yaw = util.rand_high_low(np.deg2rad(rack_cfg.PEG_YAW_LOW_HIGH))

    peg_rot1 = common.euler2rot([peg_angle, 0, 0])
    peg_rot2 = common.euler2rot([0, 0, peg_yaw])
    peg_tf1 = np.eye(4)
    peg_tf1[:-1, :-1] = peg_rot1
    peg_tf1[2, -1] = peg_base_height
    peg_tf1[1, -1] = -(peg_cyl_height / 2.0) * np.sin(peg_angle)
    peg_tf2 = np.eye(4)
    peg_tf2[:-1, :-1] = peg_rot2
    peg_tf = np.matmul(peg_tf2, peg_tf1)

    peg_cyl.apply_transform(peg_tf)
    return peg_cyl


def sample_syn_rack(rack_cfg):
    base_cyl_radius = util.rand_high_low(rack_cfg.BASE_RADIUS_LOW_HIGH)
    base_cyl_height = util.rand_high_low(rack_cfg.BASE_LENGTH_LOW_HIGH)
    base_cyl = cr.cylinder(base_cyl_radius, base_cyl_height)
    base_cyl.apply_translation([0, 0, base_cyl_height/2.0])

    peg_cyl_list = []
    if rack_cfg.N_PEGS < 0:
        n_pegs = np.random.randint(1, rack_cfg.MAX_PEGS+1) 
    else:
        n_pegs = rack_cfg.N_PEGS
    for i in range(n_pegs):
        top_bottom = 'top' if i == 0 else 'bottom'
        peg_cyl_radius = util.rand_high_low(rack_cfg.PEG_RADIUS_LOW_HIGH)
        peg_cyl_height = util.rand_high_low(rack_cfg.PEG_LENGTH_LOW_HIGH)
        peg_cyl = sample_peg(
            peg_cyl_radius, 
            peg_cyl_height, 
            base_cyl_radius, 
            base_cyl_height, 
            rack_cfg,
            top_bottom_half=top_bottom)
        peg_cyl_list.append(peg_cyl)

    rack_mesh_list = [base_cyl] + peg_cyl_list

    if rack_cfg.WITH_BOTTOM and (np.random.random() > 0.5):
        if np.random.random() > 0.5:
            bottom_cyl_radius = util.rand_high_low(rack_cfg.BOTTOM_CYLINDER_RADIUS_LOW_HIGH)
            bottom_cyl_height = util.rand_high_low(rack_cfg.BOTTOM_CYLINDER_LENGTH_LOW_HIGH)
            bottom_cyl = cr.cylinder(bottom_cyl_radius, bottom_cyl_height)
            bottom_cyl.apply_translation([0, 0, bottom_cyl_height/2.0])
            bottom_mesh = bottom_cyl
        else:
            bottom_box_side = util.rand_high_low(rack_cfg.BOTTOM_BOX_SIDE_LOW_HIGH)
            bottom_box_height = util.rand_high_low(rack_cfg.BOTTOM_BOX_HEIGHT_LOW_HIGH)
            bottom_box = cr.box([bottom_box_side, bottom_box_side, bottom_box_height])
            bottom_box.apply_translation([0, 0, bottom_box_height/2.0])
            bottom_mesh = bottom_box
        rack_mesh_list.append(bottom_mesh)

    return rack_mesh_list


def make_rack_save_occ_f(obj_name, obj_number, cfg, normalizing_factor):
    if isinstance(obj_number, int):
        obj_number = str(obj_number)
    print(f'Running for object name: {obj_name}, object number: {obj_number}')
    sample_points = get_raster_points(args.voxel_resolution)

    # create mesh
    obj_fname = osp.join(unnormalized_mesh_save_dir, obj_name + '_' + obj_number + '.obj')
    obj_mesh = trimesh.load(obj_fname, process=False)
    vertices = obj_mesh.vertices - obj_mesh.bounding_box.centroid
    norm_factor = 1 / np.max(obj_mesh.bounding_box.extents)
    vertices *= norm_factor
    obj_mesh = trimesh.Trimesh(vertices=vertices, faces=obj_mesh.faces)

    occ = inside_mesh.check_mesh_contains(obj_mesh, sample_points)
    
    # occ_pts = sample_points[np.where(occ)[0]]
    # non_occ_pts = sample_points[np.where(np.logical_not(occ))[0]]
    # util.meshcat_pcd_show(mc_vis, occ_pts, color=[255, 0, 0], name='scene/occ_pts')
    # util.meshcat_pcd_show(mc_vis, non_occ_pts, color=[255, 0, 255], name='scene/non_occ_pts')

    # save
    new_obj_name = obj_name + '_' + obj_number

    occ_save_fname = osp.join(occ_save_dir, new_obj_name + '_occupancy.npz')
    normalized_saved_obj_fname = osp.join(mesh_save_dir, new_obj_name + '.obj')

    normalized_saved_obj_fname_relative = normalized_saved_obj_fname.split(path_util.get_rndf_obj_descriptions())[1].lstrip('/')
    obj_fname_relative = obj_fname.split(path_util.get_rndf_obj_descriptions())[1].lstrip('/')

    print(f'Saving to... \nnpz file: {occ_save_fname}\nmesh_file: {normalized_saved_obj_fname_relative}')

    np.savez(
        occ_save_fname,
        mesh_fname=obj_fname_relative,
        normalized_mesh_fname=normalized_saved_obj_fname_relative,
        points=sample_points,
        occupancy=occ.reshape(-1),
        norm_factor=norm_factor
    )

    # save the normalized version of the obj file
    obj_mesh.export(normalized_saved_obj_fname)
    print(f'Done with object name: {obj_name}, object number: {obj_number}')


def main_mp(args, cfg):
    obj_name = args.obj_name
    n_objs = args.n_objs

    global_normalizing_factor = 0.0
    mp_args = [(obj_name, str(i), cfg) for i in range(n_objs)]
    for mp_arg in mp_args:
        obj_name,  obj_number, cfg = mp_arg

        # sample rack mesh
        rack_mesh_list = sample_syn_rack(rack_cfg)
        merged_rack = trimesh.util.concatenate(rack_mesh_list)
        util.meshcat_trimesh_show(mc_vis, 'scene/rack', merged_rack)
        
        # create mesh
        obj_mesh = merged_rack
        obj_mesh.vertices = obj_mesh.vertices - obj_mesh.bounding_box.centroid
        new_obj_name = obj_name + '_' + obj_number
        saved_obj_fname = osp.join(unnormalized_mesh_save_dir, new_obj_name + '.obj')
        obj_mesh.export(saved_obj_fname)
        print(f'Done with object name: {obj_name}, object number: {obj_number}')
        normalizing_factor = np.max(obj_mesh.bounding_box.extents)
        print(f'Norm factor: {normalizing_factor:.3f}')
        if normalizing_factor > global_normalizing_factor:
            global_normalizing_factor = normalizing_factor
            print(f'New norm factor: {global_normalizing_factor:.3f}')


    if args.save_occ:
        # generate the ground truth occupancy and normalized obj files
        mp_args = [(obj_name, i, cfg, global_normalizing_factor) for i in range(n_objs)]
        with Pool(args.workers) as p:
            p.starmap(make_rack_save_occ_f, mp_args)

if __name__ == '__main__':
    main_mp(args, cfg)

