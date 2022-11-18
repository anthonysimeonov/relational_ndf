from yacs.config import CfgNode as CN 

_C = CN()
_C.FOCUS_PT = [0.5, 0.0, 1.1]
_C.YAW_ANGLES = [20, 160, 210, 330]
_C.DISTANCE = 0.9
_C.PITCH = -25.0

_C.PITCH_RANGE = [0, 180]
_C.YAW_RANGE = [0, 180]
_C.DISTANCE_RANGE = [0.45, 1.2]

def get_default_cam_cfg():
    return _C.clone()