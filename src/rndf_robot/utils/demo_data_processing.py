import copy
import numpy as np

from rndf_robot.utils import util

def process_grasp_data(demo_data, query_pts):
    """Function to extract the relevant data from a demonstration
    file for a grasping skill. This involves transforming the 
    query points representing the gripper from their canonical
    pose to the 6D pose that was used for grasping (the start
    end-effector pose). 

    Args:
        demo_data (dict): [description]
        query_pts (np.ndarray): Query point set that is to be used
            for each demo in this set of skill demonstrations (passed
            as input to ensure it's the same as the point set used
            by the optimization at test time)
    """
    grasp_ee_pose_mat = util.matrix_from_pose(util.list2pose_stamped(demo_data['ee_pose_world']))
    demo_obj_pts = demo_data['object_pointcloud']
    # demo_query_pts = demo_data['gripper_pts_uniform'] 
    demo_query_pts = copy.deepcopy(query_pts)
    demo_query_pts = util.transform_pcd(demo_query_pts, grasp_ee_pose_mat)

    out_dict = {}
    out_dict['demo_query_pts'] = demo_query_pts
    out_dict['demo_obj_pts'] = demo_obj_pts
    return out_dict


def process_place_data(demo_data, query_pts):
    """Function to extract the relevant data from a demonstration
    file for a placing skill. This involves computing the relative
    transformation between the start end-effector pose, the final
    end-effector pose, and applying this transformation to the 
    initial object point cloud. The result is the object point cloud
    in it's final "placed" location relative to the query points 
    in their canonical pose. These query points represent an external
    object in the environment.

    Args:
        demo_data (dict): [description]
        query_pts (np.ndarray): Query point set that is to be used
            for each demo in this set of skill demonstrations (passed
            as input to ensure it's the same as the point set used
            by the optimization at test time)
    """
    grasp_ee_pose = util.list2pose_stamped(demo_data['start_ee_pose_world'])
    place_ee_pose = util.list2pose_stamped(demo_data['end_ee_pose_world'])
    relative_pose_mat = util.matrix_from_pose(util.get_transform(place_ee_pose, grasp_ee_pose))
    demo_obj_pts = demo_data['object_pointcloud']
    demo_obj_pts = util.transform_pcd(demo_obj_pts, relative_pose_mat)
    demo_query_pts = copy.deepcopy(query_pts)

    out_dict = {}
    out_dict['demo_query_pts'] = demo_query_pts
    out_dict['demo_obj_pts'] = demo_obj_pts

    return out_dict


def extract_grasp_query_points(demo_data, query_pts_type='uniform', **kwargs):
    """Function to extract the query points that are used for a
    grasping skill from the demonstrations. It is assumed that
    the query points to be used have already been saved in a particular
    way in the file containing the demonstration data, and can
    be directly loaded from the demo file.

    Args:
        demo_data (dict): [description]
        query_pts_type (str): Options: 'uniform', 'custom'
    """
    if query_pts_type == 'uniform':
        query_pts = demo_data['gripper_pts_uniform']
    elif query_pts_type == 'custom':
        # query_pts = demo_data['custom_gripper_query_pts']
        raise NotImplementedError
    else:
        query_pts = demo_data['gripper_pts_uniform']

    query_pts_viz = demo_data['gripper_pts']

    return query_pts, query_pts_viz


def extract_place_query_points(demo_data, query_pts_type='uniform', placement_surface='rack', gaussian_scale=1.0, **kwargs):
    """Function to extract the query points that are used for a
    grasping skill from the demonstrations. It is assumed that
    the query points to be used have already been saved in a particular
    way in the file containing the demonstration data, and can
    be directly loaded from the demo file.

    NOTE: in all cases except for "custom", we assume these points are located at the origin,
    NOT at the location of the actual "placement object" in the world. Therefore, we 
    assume a world frame pose of the "placement object" is also provided, and we transform
    these query points based on this pose such that their canonical pose in the world
    matches the location of the actual pose of this object

    Args:
        demo_data (dict): [description]
        query_pts_type (str): Options: 'uniform', 'gaussian', 'custom'
        placement_surface (str): Options: 'rack', 'table', 'shelf'
        gaussian_scale (float): Amount to scale the query point set by if we are using the gaussian points
    """
    if query_pts_type == 'custom':
        query_pts = demo_data['custom_query_points']
        query_pts_viz = demo_data['custom_query_points']
        placement_obj_pose_mat = np.eye(4)
    else:
        if placement_surface == 'rack':
            if query_pts_type == 'uniform':
                query_pts = demo_data['rack_pointcloud_uniform']
            elif query_pts_type == 'gaussian':
                query_pts = demo_data['rack_pointcloud_gaussian'] * gaussian_scale
            else:
                query_pts = demo_data['rack_pointcloud_uniform']
            
            query_pts_viz = demo_data['rack_pointcloud_gt']
            placement_obj_pose_mat = util.matrix_from_pose(util.list2pose_stamped(demo_data['rack_pose_world']))
        elif placement_surface == 'shelf':
            if query_pts_type == 'uniform':
                query_pts = demo_data['shelf_pointcloud_uniform']
            elif query_pts_type == 'gaussian':
                query_pts = demo_data['shelf_pointcloud_gaussian'] * gaussian_scale
            else:
                query_pts = demo_data['shelf_pointcloud_uniform']

            query_pts_viz = demo_data['shelf_pointcloud_gt']
            placement_obj_pose_mat = util.matrix_from_pose(util.list2pose_stamped(demo_data['shelf_pose_world']))

    if query_pts_type != 'custom':
        query_pts = util.transform_pcd(query_pts, placement_obj_pose_mat)
        query_pts_viz = util.transform_pcd(query_pts_viz, placement_obj_pose_mat)
    return query_pts, query_pts_viz

demo_data_processor_methods_dict = {
    'process_grasp_data': process_grasp_data,
    'process_place_data': process_place_data
}

demo_data_extract_query_points_methods_dict = {
    'extract_grasp_query_points': extract_grasp_query_points,
    'extract_place_query_points': extract_place_query_points
}
