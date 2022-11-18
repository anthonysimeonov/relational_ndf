import os.path as osp
import argparse
from multiprocessing import Pool
import numpy as np
import trimesh
import meshcat

from rndf_robot.utils import util, path_util
from rndf_robot.utils.mesh_util import inside_mesh
from rndf_robot.utils.mesh_util.three_util import get_raster_points

from syn_container_cfg import get_syn_container_default_cfg
from param_container import make_container

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

args = parser.parse_args()

cfg = get_syn_container_default_cfg()

mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')

# path that we will save synthesized meshes in
mesh_save_dir = osp.join(path_util.get_rndf_obj_descriptions(), args.mesh_save_dir)
unnormalized_mesh_save_dir = osp.join(path_util.get_rndf_obj_descriptions(), args.mesh_save_dir + '_unnormalized')

# path that  we will save occupancy data in
occ_save_dir = osp.join(path_util.get_rndf_obj_descriptions(), args.occ_save_dir)

util.safe_makedirs(mesh_save_dir)
util.safe_makedirs(unnormalized_mesh_save_dir)
util.safe_makedirs(occ_save_dir)


def make_container_save_occ_f(obj_name, obj_number, cfg, normalizing_factor):
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

        # sample dimensions
        bl = util.rand_high_low(cfg.BASE_LENGTH_LOW_HIGH) 
        bw = util.rand_high_low(cfg.BASE_WIDTH_LOW_HIGH)
        bt = util.rand_high_low(cfg.BASE_THICKNESS_LOW_HIGH)
        wt = util.rand_high_low(cfg.WALL_THICKNESS_LOW_HIGH)
        wh = util.rand_high_low(cfg.WALL_HEIGHT_LOW_HIGH)
        theta = util.rand_high_low(cfg.WALL_THETA_LOW_HIGH)

        # sample container
        print(f'Making container with dimensions: {bl:.3f}, {bw:.3f}, {bt:.3f}, {wt:.3f}, {wh:.3f}, {theta:.3f}')
        full_container_mesh, _ = make_container(bl=bl, bw=bw, bt=bt, wt=wt, wh=wh, th=theta, show=True, mc_vis=mc_vis)
        
        # create mesh
        obj_mesh = full_container_mesh
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
            p.starmap(make_container_save_occ_f, mp_args)

if __name__ == '__main__':
    main_mp(args, cfg)

