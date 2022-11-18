import torch


def occupancy(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['occ']
    label = (label + 1) / 2.

    loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5) + (1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    return loss_dict


def occupancy_net(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5) + (1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    return loss_dict


def occupancy_net_descriptor_dist(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    
    loss_dict['desc'] = -1 * torch.mean(torch.std(model_outputs['features'], 1))

    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5) + (1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    return loss_dict


def distance_net(model_outputs, ground_truth, val=False, delta=0.3, scaling=10.0):
    loss_dict = dict()
    label = ground_truth['sdf'].squeeze()
    clamped_label = torch.clip(label*scaling, -delta*scaling, delta*scaling)
    
    # assume that 'occ' actually corresponds to the distance, because we have not passed it through a sigmoid
    output = model_outputs['occ']
    clamped_output = torch.clip(output*scaling, -delta*scaling, delta*scaling)

    dist = torch.abs(clamped_label - clamped_output).mean()

    loss_dict['dist'] = dist

    return loss_dict

def distance_net_descriptor_dist(model_outputs, ground_truth, val=False, delta=0.3, scaling=10.0):
    loss_dict = dict()
    label = ground_truth['sdf'].squeeze()
    clamped_label = torch.clip(label*scaling, -delta*scaling, delta*scaling)
    
    # assume that 'occ' actually corresponds to the distance, because we have not passed it through a sigmoid
    output = model_outputs['occ']
    clamped_output = torch.clip(output*scaling, -delta*scaling, delta*scaling)

    dist = torch.abs(clamped_label - clamped_output).mean()

    loss_dict['dist'] = dist
    loss_dict['desc'] = -1 * torch.mean(torch.std(model_outputs['features'], 1))

    return loss_dict


def semantic(model_outputs, ground_truth, val=False):
    loss_dict = {}

    label = ground_truth['occ']
    label = ((label + 1) / 2.).squeeze()

    if val:
        loss_dict['occ'] = torch.zeros(1)
    else:
        loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'].squeeze() + 1e-5) + (1 - label) * torch.log(1 - model_outputs['occ'].squeeze() + 1e-5)).mean()

    return loss_dict
