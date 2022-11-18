# Developed based on script at
# https://github.com/marian42/shapegan/blob/master/prepare_shapenet_dataset.py

import os, os.path as osp
# Enable this when running on a computer without a screen:
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import traceback
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import trimesh
from mesh_to_sdf import get_surface_point_cloud, scale_to_unit_cube, BadMeshException

from rndf_robot.utils import util, path_util

ensure_directory = util.safe_makedirs

# import meshcat
# mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
# mc_vis['scene'].delete()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--is_shapenet', action='store_true', help='If these are shapenet models')
parser.add_argument('--dataset_name', type=str, required=True, help='Name of the folder to save SDF data inside')
parser.add_argument('--models_directory', type=str, required=True, help='Path to folder with .obj files')

args = parser.parse_args()

IS_SHAPENET = args.is_shapenet
DATASET_NAME = osp.join(path_util.get_rndf_obj_descriptions(), 'sdf', args.dataset_name)
DIRECTORY_MODELS = osp.join(path_util.get_rndf_obj_descriptions(), args.models_directory)

MODEL_EXTENSION = '.obj'
DIRECTORY_VOXELS = f'{DATASET_NAME}/voxels_{{:d}}/'
DIRECTORY_UNIFORM = f'{DATASET_NAME}/uniform/'
DIRECTORY_SURFACE = f'{DATASET_NAME}/surface/'
DIRECTORY_SDF_CLOUD = f'{DATASET_NAME}/cloud/'
DIRECTORY_BAD_MESHES = f'{DATASET_NAME}/bad_meshes/'

# Voxel resolutions to create.
# Set to [] if no voxels are needed.
# Set to [32] for for all models except for the progressively growing DeepSDF/Voxel GAN
# VOXEL_RESOLUTIONS = [8, 16, 32, 64]
VOXEL_RESOLUTIONS = []

CREATE_SDF_CLOUDS = True # For DeepSDF autodecoder, contains uniformly and non-uniformly sampled points as proposed in the DeepSDF paper
CREATE_UNIFORM_AND_SURFACE = False # Uniformly sampled points for the Pointnet-based GAN and surface point clouds for the pointnet-based GAN with refinement

SDF_POINT_CLOUD_SIZE = 200000 # For DeepSDF point clouds (CREATE_SDF_CLOUDS)
POINT_CLOUD_SAMPLE_SIZE = 64**3 # For uniform and surface points (CREATE_UNIFORM_AND_SURFACE)

# Options for virtual scans used to generate SDFs
USE_DEPTH_BUFFER = True
SCAN_COUNT = 50
SCAN_RESOLUTION = 1024

def get_model_files():
    for directory, _, files in os.walk(DIRECTORY_MODELS):
        for filename in files:
            if filename.endswith(MODEL_EXTENSION):
                yield os.path.join(directory, filename)

def get_hash(filename):
    return filename.split('/')[-3]

def get_hash_non_shapenet(filename):
    return filename.split('/')[-1].split('.obj')[0]

def get_voxel_filename(model_filename, resolution):
    return os.path.join(DIRECTORY_VOXELS.format(resolution), get_hash(model_filename) + '.npy')

def get_uniform_filename(model_filename):
    return os.path.join(DIRECTORY_UNIFORM, get_hash(model_filename) + '.npy')

def get_surface_filename(model_filename):
    return os.path.join(DIRECTORY_SURFACE, get_hash(model_filename) + '.npy')

def get_sdf_cloud_filename(model_filename, shapenet=True):
    if shapenet:
        return os.path.join(DIRECTORY_SDF_CLOUD, get_hash(model_filename) + '.npy')
    else:
        return os.path.join(DIRECTORY_SDF_CLOUD, get_hash_non_shapenet(model_filename) + '.npy')

def get_bad_mesh_filename(model_filename, shapenet=True):
    if shapenet:
        return os.path.join(DIRECTORY_BAD_MESHES, get_hash(model_filename))
    else:
        return os.path.join(DIRECTORY_BAD_MESHES, get_hash_non_shapenet(model_filename))

def mark_bad_mesh(model_filename):
    filename = get_bad_mesh_filename(model_filename, shapenet=IS_SHAPENET)
    ensure_directory(os.path.dirname(filename))            
    open(filename, 'w').close()

def is_bad_mesh(model_filename):
    return os.path.exists(get_bad_mesh_filename(model_filename, shapenet=IS_SHAPENET))

def get_uniform_and_surface_points(surface_point_cloud, number_of_points = 200000):
    unit_sphere_points = np.random.uniform(-1, 1, size=(number_of_points * 2, 3)).astype(np.float32)
    unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]
    uniform_points = unit_sphere_points[:number_of_points, :]

    distances, indices = surface_point_cloud.kd_tree.query(uniform_points)
    uniform_sdf = distances.astype(np.float32).reshape(-1) * -1
    uniform_sdf[surface_point_cloud.is_outside(uniform_points)] *= -1

    surface_points = surface_point_cloud.points[indices[:, 0], :]
    near_surface_points = surface_points + np.random.normal(scale=0.0025, size=surface_points.shape).astype(np.float32)
    near_surface_sdf = surface_point_cloud.get_sdf(near_surface_points, use_depth_buffer=USE_DEPTH_BUFFER)
    
    model_size = np.count_nonzero(uniform_sdf < 0) / number_of_points
    if model_size < 0.01:
        raise BadMeshException()

    return uniform_points, uniform_sdf, near_surface_points, near_surface_sdf

def process_model_file(filename):
    try:
        if is_bad_mesh(filename):
            print('Skipping bad mesh (is_bad_mesh): {:s}'.format(get_hash_non_shapenet(filename)))
            return
        
        mesh = trimesh.load(filename)

        voxel_filenames = [get_voxel_filename(filename, resolution) for resolution in VOXEL_RESOLUTIONS]
        if not all(os.path.exists(f) for f in voxel_filenames):
            mesh_unit_cube = scale_to_unit_cube(mesh)
            surface_point_cloud = get_surface_point_cloud(mesh_unit_cube, bounding_radius=3**0.5, scan_count=SCAN_COUNT, scan_resolution=SCAN_RESOLUTION)
            try:
                for resolution in VOXEL_RESOLUTIONS:
                    voxels = surface_point_cloud.get_voxels(resolution, use_depth_buffer=USE_DEPTH_BUFFER, check_result=True)
                    np.save(get_voxel_filename(filename, resolution), voxels)
                    del voxels
            
            except BadMeshException:
                tqdm.write("Skipping bad mesh. ({:s})".format(get_hash(filename)))
                mark_bad_mesh(filename)
                return
            del mesh_unit_cube, surface_point_cloud
        

        create_uniform_and_surface = CREATE_UNIFORM_AND_SURFACE and (not os.path.exists(get_uniform_filename(filename)) or not os.path.exists(get_surface_filename(filename)))
        create_sdf_clouds = CREATE_SDF_CLOUDS and not os.path.exists(get_sdf_cloud_filename(filename, shapenet=IS_SHAPENET))

        if create_uniform_and_surface or create_sdf_clouds:

            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump().sum()

            vertices = mesh.vertices - mesh.bounding_box.centroid
            distances = np.linalg.norm(vertices, axis=1)
            vertices /= np.max(distances)

            mesh_unit_sphere = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
            norm_factor = np.max(distances)

            surface_point_cloud = get_surface_point_cloud(mesh_unit_sphere, bounding_radius=1, scan_count=SCAN_COUNT, scan_resolution=SCAN_RESOLUTION)
            try:
                if create_uniform_and_surface:
                    uniform_points, uniform_sdf, near_surface_points, near_surface_sdf = get_uniform_and_surface_points(surface_point_cloud, number_of_points=POINT_CLOUD_SAMPLE_SIZE)
                    
                    combined_uniform = np.concatenate((uniform_points, uniform_sdf[:, np.newaxis]), axis=1)
                    np.save(get_uniform_filename(filename), combined_uniform)

                    combined_surface = np.concatenate((near_surface_points, near_surface_sdf[:, np.newaxis]), axis=1)
                    np.save(get_surface_filename(filename), combined_surface)

                if create_sdf_clouds:
                    sdf_points, sdf_values = surface_point_cloud.sample_sdf_near_surface(use_scans=True, sign_method='depth' if USE_DEPTH_BUFFER else 'normal', number_of_points=SDF_POINT_CLOUD_SIZE, min_size=0.015)
                    combined = np.concatenate((sdf_points, sdf_values[:, np.newaxis]), axis=1)
                    sdf_cloud_fname = get_sdf_cloud_filename(filename, shapenet=IS_SHAPENET).replace('.npy', '.npz')
                    print(f'Saving SDF cloud to file: {sdf_cloud_fname}')
                    np.savez(sdf_cloud_fname, coords_sdf=combined, norm_factor=norm_factor)
            except BadMeshException:
                if IS_SHAPENET:
                    tqdm.write("Skipping bad mesh. ({:s})".format(get_hash(filename)))
                else:
                    tqdm.write("Skipping bad mesh. ({:s})".format(get_hash_non_shapenet(filename)))
                mark_bad_mesh(filename)
                return
            del mesh_unit_sphere, surface_point_cloud
            
    except:
        traceback.print_exc()


def process_model_files():
    for res in VOXEL_RESOLUTIONS:
        ensure_directory(DIRECTORY_VOXELS.format(res))
    if CREATE_UNIFORM_AND_SURFACE:
        ensure_directory(DIRECTORY_UNIFORM)
        ensure_directory(DIRECTORY_SURFACE)
    if CREATE_SDF_CLOUDS:
        ensure_directory(DIRECTORY_SDF_CLOUD)
    ensure_directory(DIRECTORY_BAD_MESHES)

    files = list(get_model_files())
    print(f"Models directory: {DIRECTORY_MODELS}")
    print(f"Model files (total: {len(files)})")
    print('\n'.join(files))

    worker_count = os.cpu_count() // 2
    print("Using {:d} processes.".format(worker_count))
    pool = Pool(worker_count)

    progress = tqdm(total=len(files))
    def on_complete(*_):
        progress.update()

    for filename in files:
        pool.apply_async(process_model_file, args=(filename,), callback=on_complete)
    pool.close()
    pool.join()

def combine_sdf_clouds():
    import torch
    print("Combining SDF point clouds...")

    files = list(sorted(get_model_files()))
    files = [f for f in files if os.path.exists(get_sdf_cloud_filename(f, shapenet=IS_SHAPENET))]
    
    N = len(files)
    points = torch.zeros((N * SDF_POINT_CLOUD_SIZE, 3))
    sdf = torch.zeros((N * SDF_POINT_CLOUD_SIZE))
    position = 0

    for file_name in tqdm(files):
        numpy_array = np.load(get_sdf_cloud_filename(file_name, shapenet=IS_SHAPENET))
        points[position * SDF_POINT_CLOUD_SIZE:(position + 1) * SDF_POINT_CLOUD_SIZE, :] = torch.tensor(numpy_array[:, :3])
        sdf[position * SDF_POINT_CLOUD_SIZE:(position + 1) * SDF_POINT_CLOUD_SIZE] = torch.tensor(numpy_array[:, 3])
        del numpy_array
        position += 1
    
    print("Saving combined SDF clouds...")
    torch.save(points, os.path.join(DIRECTORY_SDF_CLOUD, 'sdf_points.to'))
    torch.save(sdf, os.path.join(DIRECTORY_SDF_CLOUD, 'sdf_values.to'))

if __name__ == '__main__':
    process_model_files()
    if CREATE_SDF_CLOUDS:
        combine_sdf_clouds()
