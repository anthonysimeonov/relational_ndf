import os, os.path as osp
import sys
import random
import numpy as np
import torch
import copy
import trimesh

from airobot import log_info, log_warn, log_debug, log_critical

from rndf_robot.utils import util
from rndf_robot.opt.optimizer import OccNetOptimizer


def infer_relation_intersection(mc_vis, parent_optimizer, child_optimizer, parent_target_desc, child_target_desc, 
                                parent_pcd, child_pcd, parent_query_points, child_query_points, opt_visualize=False, visualize=False,
                                *args, **kwargs):
    out_parent_feat = parent_optimizer.optimize_transform_implicit(parent_pcd, ee=True, return_score_list=True, return_final_desc=True, target_act_hat=parent_target_desc, visualize=opt_visualize)
    parent_feat_pose_mats, best_parent_idx, desc_dist_parent, desc_parent = out_parent_feat

    parent_feat_pose_mat = parent_feat_pose_mats[best_parent_idx]
    child_query_points = util.transform_pcd(parent_query_points, parent_feat_pose_mats[best_parent_idx])
    child_optimizer.set_query_points(child_query_points)

    # now we want to do the same thing relative to the child objects
    out_child_feat = child_optimizer.optimize_transform_implicit(child_pcd, ee=False, return_score_list=True, return_final_desc=True, target_act_hat=child_target_desc, visualize=opt_visualize)
    child_feat_pose_mats, best_child_idx, desc_dist_child, desc_child = out_child_feat

    child2parent_feat_pose_mat = child_feat_pose_mats[best_child_idx]
    parent2child_feat_pose_mat = np.linalg.inv(child2parent_feat_pose_mat)
    child_feat_pose_mat = np.matmul(parent2child_feat_pose_mat, parent_feat_pose_mat)
    transformed_child_feat_pose_mat = np.matmul(child2parent_feat_pose_mat, child_feat_pose_mat)

    if visualize:
        util.meshcat_frame_show(mc_vis, 'scene/parent_feat_pose', parent_feat_pose_mat)
        util.meshcat_frame_show(mc_vis, 'scene/child_feat_pose', child_feat_pose_mat)
        util.meshcat_frame_show(mc_vis, 'scene/transformed_child_feat_pose', transformed_child_feat_pose_mat)
    
    # finally, the relative pose we should execute is here
    relative_transformation = child_feat_pose_mats[best_child_idx]
    return relative_transformation


def filter_parent_child_pcd(pf_pcd, cf_pcd):
    # filter out some noisy points

    # pf_inliers = np.where(pf_pcd[:, 2] > 0.025)[0]
    # pf_pcd = pf_pcd[pf_inliers]

    # pf_mean = np.mean(pf_pcd, axis=0)
    # pf_inliers = np.where(np.linalg.norm(pf_pcd[:, :-1] - pf_mean[:-1], 2, 1) < 0.2)[0]
    # pf_pcd = pf_pcd[pf_inliers]

    pf_mean = np.mean(pf_pcd, axis=0)
    pf_inliers = np.where(np.linalg.norm(pf_pcd - pf_mean, 2, 1) < 0.2)[0]
    pf_pcd = pf_pcd[pf_inliers]

    cf_mean = np.mean(cf_pcd, axis=0)
    cf_inliers = np.where(np.linalg.norm(cf_pcd - cf_mean, 2, 1) < 0.2)[0]
    # cf_inliers = np.where(np.linalg.norm(cf_pcd - cf_mean, 2, 1) < 0.1)[0]
    cf_pcd = cf_pcd[cf_inliers]
    return pf_pcd, cf_pcd


def keypoint_offset(cf_pcd, offset=0.025, type='bottom'):
    assert type in ['bottom', 'top', 'mean'], 'Invalid keypoint offset type!'
    keypoint_trans = np.mean(cf_pcd, axis=0)

    # this assumes top/bottom are aligned with the z-axis! 
    if type == 'bottom':
        keypoint_trans[2] = np.min(cf_pcd[:, 2])
        keypoint_trans[2] -= offset
    elif type == 'top':
        keypoint_trans[2] = np.max(cf_pcd[:, 2])
        keypoint_trans[2] += offset

    return keypoint_trans


def create_target_descriptors(parent_model, child_model, pc_demo_dict, target_desc_fname, 
                              cfg, query_scale=0.025, opt_iterations=650, 
                              scale_pcds=False, manual_kp_adjustment=False,
                              alignment_rounds=3, target_rounds=1, pc_reference='parent', 
                              skip_alignment=False, n_demos='all', manual_target_idx=-1,
                              add_noise=False, interaction_pt_noise_std=0.01, 
                              use_keypoint_offset=False, keypoint_offset_params=None,
                              visualize=False, mc_vis=None):
    """
    Create a .npz file containing information about a relational multi-object
    task. Will create target pose descriptors for parent object and child
    object, and save the set of query points that should be used at test 
    time to infer poses of local part geometry for both parent and child
    object

    Args:
        parent_model (VNNOccNet): Parent NDF model, with weights loaded
        child_model (VNNOccNet): Child NDF model, with weights loaded
        pc_demo_dict (dict): Keys: ['parent', 'child'], values are dicts. Lower-level
            dicts with keys: ['demo_final_pcds'], values are final point clouds from
            the demonstrations for parent/child object, respectively
        target_desc_fname (str): Name of file to save descriptors to
        cfg (yacs.CfgNode): Contains config params for OccNetOptimizers
        query_scale (float): Scale of Gaussian distributed query points at origin
        opt_iterations (int): Number of OccNetOptimizer iterations to use
        scale_pcds (bool): If True, apply a larger scaling to the point clouds in the
            demos to encourage some intersections between the bounding boxes
        manual_kp_adjustment (bool): If True, will allow for manual keypoint position 
            adjustment
        alignment_rounds (int): Number of full rounds to go through all the demos, for 
            a randomly selected target demo
        target_rounds (int): Number of times to select a random target demo
        pc_reference (str): 'parent' or 'child', which object to use in alignment
        skip_alignment (bool): If True, don't perform alignment of the orientations,
            just use the estimated positions using our intersection heuristic
        n_demos (str or int): If 'all' use all, else use the integer number
        interaction_pt_noise_std (float): Standard deviation of noise to add for 
            noisy keypoint position experiment
        use_keypoint_offset (bool): If True, don't use the bounding box intersection method.
            Instead, use a manually specified offset relative to a point on the object
        keypoint_offset_params (dict): Contains keyword arguments for passing to the
            function which initializes the keypoint position using an offset from one 
            of the points on the object.
        visualize (bool): If True, use meshcat to visualize what's going on while building
            target descriptors. meshcat.Visualizer handler must be passed in if using visualization
        mc_vis (meshcat.Visualzer): meshcat handler
    """
    assert not (visualize and (mc_vis is None)), 'mc_vis cannot be None if visualize=True'

    target_desc_folder = '/'.join(target_desc_fname.split('/')[:-2])
    if osp.exists(target_desc_fname):
        import datetime
        import shutil
        nowstr = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        # save a copy of the current target desc before we overwrite it
        backup_desc_folder = '/'.join(target_desc_fname.split('/')[:-1])
        backup_desc_folder = osp.join(backup_desc_folder, nowstr) 
        util.safe_makedirs(backup_desc_folder)
        log_warn(f'\n\n[create_target_descriptors] \nFilename: {target_desc_fname} already exists! \n\nSaving to folder: {backup_desc_folder} as a backup before overwriting\n\n')

        backup_desc_fname = osp.join(backup_desc_folder, target_desc_fname.split('/')[-1])
        shutil.copy(target_desc_fname, backup_desc_fname) 

        assert osp.exists(backup_desc_fname), f'Something went wrong with backing up the descriptors to path: {backup_desc_fname}!'

    # initialize set of world frame query points with specified variance
    parent_query_points = np.random.normal(scale=query_scale, size=(500, 3))
    child_query_points = copy.deepcopy(parent_query_points)

    # create optimizers that will be used for alignment
    parent_optimizer = OccNetOptimizer(
        parent_model,
        query_pts=parent_query_points,
        query_pts_real_shape=parent_query_points,
        opt_iterations=opt_iterations,
        cfg=cfg.OPTIMIZER)

    child_optimizer = OccNetOptimizer(
        child_model,
        query_pts=child_query_points,
        query_pts_real_shape=child_query_points,
        opt_iterations=opt_iterations,
        cfg=cfg.OPTIMIZER)

    pc_demo_dict['parent']['query_pts'] = []
    pc_demo_dict['child']['query_pts'] = []

    # one heuristic we used for creating keypoint labels is to estimate where
    # the bounding boxes of the two shapes intersect
    # another is to use a fixed offset away from a known task-specific point,
    # such as the bottom of the bottle
    # here, we are running this for each demo to get a candidate keypoint

    # prepare list of keypoint translations
    trans_list = []
    valid_targets = []

    visualize_bounding_box_intersection = False  # can set to true for extra visualizations
    n_demos = len(pc_demo_dict['parent']['demo_final_pcds']) if n_demos == 'all' else n_demos
    demo_idxs = np.random.permutation(len(pc_demo_dict['parent']['demo_final_pcds']))[:n_demos]
    for i in range(len(pc_demo_dict['parent']['demo_final_pcds'])):
        pf_pcd = pc_demo_dict['parent']['demo_final_pcds'][i]
        cf_pcd = pc_demo_dict['child']['demo_final_pcds'][i]

        # # scale these up a little bit
        if scale_pcds:
            pf_mean, cf_mean = np.mean(pf_pcd, axis=0), np.mean(cf_pcd, axis=0)
            pf_pcd -= pf_mean; pf_pcd *= 1.25; pf_pcd += pf_mean
            cf_pcd -= cf_mean; cf_pcd *= 1.25; cf_pcd += cf_mean

        # get rid of some outlier artifacts that sometimes show up
        pf_pcd, cf_pcd = filter_parent_child_pcd(pf_pcd, cf_pcd)

        if visualize:
            util.meshcat_pcd_show(mc_vis, cf_pcd, color=[0, 0, 255], name=f'scene/{i}/final_pcds_child_{i}')
            util.meshcat_pcd_show(mc_vis, pf_pcd, color=[255, 0, 0], name=f'scene/{i}/final_pcds_parent_{i}')

        if len(valid_targets) > 0:
            print('Got at least one valid target, continuing')
            continue

        # get bounding boxes and estimate a point where they intersect
        pbb = trimesh.PointCloud(pf_pcd).bounding_box_oriented
        cbb = trimesh.PointCloud(cf_pcd).bounding_box_oriented

        pbb_pts = pbb.sample_volume(1000)
        cbb_pts = cbb.sample_volume(1000)

        pbb_mask = pbb.contains(cbb_pts)
        cbb_mask = cbb.contains(pbb_pts)

        bb_pts = np.concatenate([pbb_pts[cbb_mask], cbb_pts[pbb_mask]], axis=0)

        if bb_pts.shape[0] > 4:
            fine_grained_bb = trimesh.PointCloud(bb_pts).bounding_box_oriented.to_mesh()

            intersection_trans = fine_grained_bb.centroid

            interaction_noise = None
            if add_noise:
                interaction_noise = np.random.normal(scale=interaction_pt_noise_std)
                log_warn(f'Adding interaction noise! Value: {interaction_noise}')
                intersection_trans = copy.deepcopy(intersection_trans) + interaction_noise

            if use_keypoint_offset:
                intersection_trans = keypoint_offset(cf_pcd, **keypoint_offset_params)

            if visualize_bounding_box_intersection:
                util.meshcat_pcd_show(mc_vis, bb_pts, color=[0, 255, 0], name=f'scene/{i}/final_pcds_bb_pts_{i}')
                util.meshcat_pcd_show(mc_vis, pf_pcd, color=[255, 0, 0], name=f'scene/{i}/final_pcds_parent_{i}')
                util.meshcat_pcd_show(mc_vis, cf_pcd, color=[0, 0, 255], name=f'scene/{i}/final_pcds_child_{i}')
                util.meshcat_trimesh_show(mc_vis, f'scene/{i}/final_bb_pts_bb_{i}', fine_grained_bb, opacity=0.4)

                sph = trimesh.creation.uv_sphere(0.005).apply_translation(intersection_trans)
                util.meshcat_trimesh_show(mc_vis, f'scene/{i}/intersection_trans', sph, (255, 0, 0))

            delta = 0.0025
            if manual_kp_adjustment:
                log_info(f'Manual keypoint adjustment for reference query points')
                kp_trans = copy.deepcopy(intersection_trans)
                kp_trans_original = copy.deepcopy(kp_trans)
                while True:
                    adj_key = input('Press "w/s to adjust x, a/d to adjust y, and q/e to adjust z\nPress z to restart\nPress x to exit \n')

                    if adj_key not in ['w', 's', 'a', 'd', 'q', 'e', 'z', 'x']:
                        log_info(f'Unrecognized key: {adj_key}')
                        continue

                    if adj_key == 'x':
                        break
                
                    if adj_key == 'z':
                        kp_trans = copy.deepcopy(kp_trans_original)
                    
                    adj_x = 0.0 if adj_key not in ['w', 's'] else delta
                    adj_x = -1.0 * adj_x if adj_key == 'w' else adj_x

                    adj_y = 0.0 if adj_key not in ['a', 'd'] else delta
                    adj_y = -1.0 * adj_y if adj_key == 'a' else adj_y

                    adj_z = 0.0 if adj_key not in ['q', 'e'] else delta
                    adj_z = -1.0 * adj_z if adj_key == 'q' else adj_z
                
                    adj_vec = np.array([adj_x, adj_y, adj_z])

                    kp_trans = kp_trans + adj_vec
                    sph = trimesh.creation.uv_sphere(0.005).apply_translation(kp_trans)
                    util.meshcat_trimesh_show(mc_vis, f'scene/{i}/keypoint_trans', sph, (0, 255, 0))
            
                intersection_trans = kp_trans

            trans_list.append(intersection_trans)

            # translate the query points to this position
            demo_parent_qp = parent_query_points + intersection_trans
            demo_child_qp = child_query_points + intersection_trans

            pc_demo_dict['parent']['query_pts'].append(demo_parent_qp)
            pc_demo_dict['child']['query_pts'].append(demo_child_qp)

            valid_targets.append(i)

            if visualize_bounding_box_intersection:
                util.meshcat_pcd_show(mc_vis, demo_parent_qp, color=[255, 255, 0], name=f'scene/{i}/translate_parent_qp_{i}')
                util.meshcat_pcd_show(mc_vis, demo_child_qp, color=[0, 255, 255], name=f'scene/{i}/translate_child_qp_{i}')

        else:
            pc_demo_dict['parent']['query_pts'].append(None)
            pc_demo_dict['child']['query_pts'].append(None)
            trans_list.append(None)
            log_warn(f'No bounding box intersection for demo number: {i}')

    if visualize:
        parent_optimizer.setup_meshcat(mc_vis)
        child_optimizer.setup_meshcat(mc_vis)
    parent_target_desc_list = []
    child_target_desc_list = []

    if add_noise:
        valid_targets = [random.sample(valid_targets, 1)[0]]  # cycle through the demos for the noise experiment
    else:
        if manual_target_idx < 0:
            valid_targets = [0]  # Just use the first demonstration (arbitray choice!)
        else:
            assert manual_target_idx < len(valid_targets), 'Manually specified demonstration index too large for number of (valid) demos!'
            valid_targets = [manual_target_idx]

    parent_descriptor_variance_list = []
    child_descriptor_variance_list = []

    # loop through all as the targets
    for target_idx in valid_targets:
        # target_idx = valid_targets[-2]
        parent_pcd_target = pc_demo_dict['parent']['demo_final_pcds'][target_idx]
        child_pcd_target = pc_demo_dict['child']['demo_final_pcds'][target_idx]

        #########################################################################

        parent_pcd_target, child_pcd_target = filter_parent_child_pcd(parent_pcd_target, child_pcd_target)

        #########################################################################

        # qp == query points
        # set up torch tensors with query points and initial positions
        qp_target = torch.from_numpy(pc_demo_dict['parent']['query_pts'][target_idx]).float().reshape((1, -1, 3)).cuda()
        qp_target_frame_pose = np.eye(4); qp_target_frame_pose[:-1, -1] = trans_list[target_idx]
        qp_target_np = qp_target.detach().cpu().numpy().reshape(-1, 3)

        qp_target_c = torch.from_numpy(pc_demo_dict['child']['query_pts'][target_idx]).float().reshape((1, -1, 3)).cuda()
        qp_target_c_frame_pose = np.eye(4); qp_target_c_frame_pose[:-1, -1] = trans_list[target_idx]
        qp_target_c_np = qp_target_c.detach().cpu().numpy().reshape(-1, 3)

        # get target descriptors that correspond to these query points
        parent_target_desc = parent_optimizer.get_pose_descriptor(parent_pcd_target, qp_target_frame_pose)
        parent_target_desc_orig = parent_target_desc.clone().detach()

        child_target_desc = child_optimizer.get_pose_descriptor(child_pcd_target, qp_target_c_frame_pose)
        child_target_desc_orig = child_target_desc.clone().detach()

        if skip_alignment:
            parent_target_desc_list.append(parent_target_desc.detach())
            child_target_desc_list.append(child_target_desc.detach())
            log_warn('\n\nSkipping alignment between demos! Just using the pose descriptor interaction point\n\n')
            continue

        sz = parent_target_desc_orig.size()

        # prepare list of final descriptors to average (updates on each iter based on if new loss is better than old loss)
        parent_last_outdesc = [None] * len(pc_demo_dict['parent']['demo_final_pcds'])
        parent_last_outloss = [np.inf] * len(pc_demo_dict['parent']['demo_final_pcds'])

        child_last_outdesc = [None] * len(pc_demo_dict['child']['demo_final_pcds'])
        child_last_outloss = [np.inf] * len(pc_demo_dict['child']['demo_final_pcds'])
    
        if visualize:
            util.meshcat_pcd_show(mc_vis, parent_pcd_target, color=[255, 0, 0], name=f'scene/{target_idx}/final_pcds_parent_{target_idx}')
            util.meshcat_pcd_show(mc_vis, child_pcd_target, color=[0, 0, 255], name=f'scene/{target_idx}/final_pcds_child_{target_idx}')
            util.meshcat_pcd_show(mc_vis, qp_target_np, color=[0, 0, 0], name='scene/qp_target')
            util.meshcat_frame_show(mc_vis, 'scene/qp_target_frame', qp_target_frame_pose)
            util.meshcat_pcd_show(mc_vis, qp_target_c_np, color=[50, 0, 50], name='scene/qp_target_c')
            util.meshcat_frame_show(mc_vis, 'scene/qp_target_c_frame', qp_target_c_frame_pose)

        if n_demos == 1:
            parent_target_desc_list.append(parent_target_desc)
            child_target_desc_list.append(child_target_desc)
            log_warn('\n\nUsing just 1 demo! Breaking before alignment!\n\n')
            break

        if skip_alignment:
            parent_target_desc_list = parent_target_desc_list[:n_demos]
            child_target_desc_list = child_target_desc_list[:n_demos]
            log_warn('\n\nskip_alignment set to True! Breaking before alignment!\n\n')
            break

        for it in range(alignment_rounds):
            
            for idx in demo_idxs:
                print(f'\n\nAligning demo number: {idx} to target demo number: {target_idx}\n\n')
                parent_pcd = pc_demo_dict['parent']['demo_final_pcds'][idx]
                child_pcd = pc_demo_dict['child']['demo_final_pcds'][idx]

                parent_pcd, child_pcd = filter_parent_child_pcd(parent_pcd, child_pcd)

                if idx == target_idx:
                    if it < 1:
                        parent_last_outdesc[idx] = parent_target_desc_orig
                        child_last_outdesc[idx] = child_target_desc_orig
                        continue
                    else:
                        print('Updating original target idx')
                        pass

                if pc_reference == 'parent':
                    # optimize each
                    parent_out_tf, parent_out_best_idx, parent_out_losses, parent_out_descs = parent_optimizer.optimize_transform_implicit(parent_pcd, target_act_hat=parent_target_desc, return_score_list=True, return_final_desc=True, visualize=visualize)

                    if parent_out_losses[parent_out_best_idx] < parent_last_outloss[idx]:
                        print(f'Parent Target: {target_idx}, Updating best loss {idx}: last: {parent_last_outloss[idx]:.5f}, new: {parent_out_losses[parent_out_best_idx]:.5f}')
                        parent_last_outloss[idx] = parent_out_losses[parent_out_best_idx]
                        parent_last_outdesc[idx] = parent_out_descs[parent_out_best_idx].view(sz)

                        parent_out_tf_best = parent_out_tf[parent_out_best_idx]
                        parent_out_qp = util.transform_pcd(parent_query_points, parent_out_tf_best)

                        child_out_desc = child_optimizer.get_pose_descriptor(child_pcd, parent_out_tf_best).detach()
                        child_last_outdesc[idx] = child_out_desc

                        out_child_sanity = child_optimizer.optimize_transform_implicit(child_pcd, target_act_hat=child_target_desc, return_score_list=True, return_final_desc=True, visualize=visualize)

                        if visualize:
                            util.meshcat_frame_show(mc_vis, f'scene/out_{idx}_tf_best_parent', parent_out_tf_best)
                            util.meshcat_pcd_show(mc_vis, parent_out_qp, color=[255, 0, 255], name=f'scene/out_{idx}_qp_parent')
                else:
                    # optimize each
                    child_out_tf, child_out_best_idx, child_out_losses, child_out_descs = child_optimizer.optimize_transform_implicit(child_pcd, target_act_hat=child_target_desc, return_score_list=True, return_final_desc=True, visualize=visualize)

                    if child_out_losses[child_out_best_idx] < child_last_outloss[idx]:
                        print(f'Parent Target: {target_idx}, Updating best loss {idx}: last: {child_last_outloss[idx]:.5f}, new: {child_out_losses[child_out_best_idx]:.5f}')
                        child_last_outloss[idx] = child_out_losses[child_out_best_idx]
                        child_last_outdesc[idx] = child_out_descs[child_out_best_idx].view(sz)

                        child_out_tf_best = child_out_tf[child_out_best_idx]
                        child_out_qp = util.transform_pcd(child_query_points, child_out_tf_best)

                        parent_out_desc = parent_optimizer.get_pose_descriptor(parent_pcd, child_out_tf_best).detach()
                        parent_last_outdesc[idx] = parent_out_desc

                        out_parent_sanity = parent_optimizer.optimize_transform_implicit(parent_pcd, target_act_hat=parent_target_desc, return_score_list=True, return_final_desc=True, visualize=visualize)

                        if visualize:
                            util.meshcat_frame_show(mc_vis, f'scene/out_{idx}_tf_best_child', child_out_tf_best)
                            util.meshcat_pcd_show(mc_vis, child_out_qp, color=[255, 0, 255], name=f'scene/out_{idx}_qp_child')

            # get new target descriptor
            if it < 1:
                parent_target_stack = torch.stack([parent_target_desc, parent_target_desc_orig] + parent_last_outdesc, 0)
                child_target_stack = torch.stack([child_target_desc, child_target_desc_orig] + child_last_outdesc, 0)
            else:
                parent_target_stack = torch.stack([parent_target_desc] + parent_last_outdesc, 0)
                child_target_stack = torch.stack([child_target_desc] + child_last_outdesc, 0)

            parent_target_desc = torch.mean(parent_target_stack, 0).detach()
            child_target_desc = torch.mean(child_target_stack, 0).detach()

            parent_var = torch.var(parent_target_stack, 0).mean().detach().item()
            child_var = torch.var(child_target_stack, 0).mean().detach().item()
            parent_descriptor_variance_list.append(parent_var)
            child_descriptor_variance_list.append(child_var)

        parent_target_desc_list.append(parent_target_desc)
        child_target_desc_list.append(child_target_desc)
    
    # take an average over the full set to get the final target descriptors for parent and child objects
    parent_overall_target_desc = torch.mean(torch.stack(parent_target_desc_list, 0), 0)
    child_overall_target_desc = torch.mean(torch.stack(child_target_desc_list, 0), 0)

    parent_descriptor_variance = np.asarray(parent_descriptor_variance_list)
    child_descriptor_variance = np.asarray(child_descriptor_variance_list)

    print(f'Done! Saving to {target_desc_fname}')
    np.savez(
        target_desc_fname,
        parent_overall_target_desc=parent_overall_target_desc.detach().cpu().numpy(),
        child_overall_target_desc=child_overall_target_desc.detach().cpu().numpy(),
        parent_query_points=parent_query_points,
        parent_descriptor_variance=parent_descriptor_variance,
        child_descriptor_variance=child_descriptor_variance,
        add_noise=add_noise,
        interaction_pt_noise_std=interaction_pt_noise_std,
        interaction_pt_noise_value=interaction_noise
    )
