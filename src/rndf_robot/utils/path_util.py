import os, os.path as osp


def get_rndf_src():
    return os.environ['RNDF_SOURCE_DIR']


def get_rndf_config():
    return osp.join(get_rndf_src(), 'config')


def get_rndf_share():
    return osp.join(get_rndf_src(), 'share')


def get_rndf_data():
    return osp.join(get_rndf_src(), 'data')


def get_rndf_recon_data():
    return osp.join(get_rndf_src(), 'data_gen/data')


def get_rndf_eval_data():
    return osp.join(get_rndf_src(), 'eval_data')


def get_rndf_descriptions():
    return osp.join(get_rndf_src(), 'descriptions')


def get_rndf_obj_descriptions():
    return osp.join(get_rndf_descriptions(), 'objects')


def get_rndf_demo_obj_descriptions():
    return osp.join(get_rndf_descriptions(), 'demo_objects')


def get_rndf_assets():
    return osp.join(get_rndf_src(), 'assets')


def get_rndf_model_weights():
    return osp.join(get_rndf_src(), 'model_weights')


def get_conv_onet_src():
    assert 'CONV_ONET_SOURCE_DIR' in os.environ.keys(), 'CONV_ONET_SOURCE_DIR not found in environment variables'
    return os.environ['CONV_ONET_SOURCE_DIR']
