"""Measured torso inclination for bow completion, independent of leg motion."""
import mujoco
import numpy as np
_model=None
def torso_pitch(q):
    global _model
    if _model is None:
        from paths import ROOT
        _model=mujoco.MjModel.from_xml_path(str(ROOT/'GR00T-WholeBodyControl/gear_sonic_deploy/g1/scene_29dof.xml'))
    data=mujoco.MjData(_model);body=mujoco.mj_name2id(_model,mujoco.mjtObj.mjOBJ_BODY,'torso_link');assert body>=0
    result=[]
    for frame in q:
        data.qpos[:]=frame;mujoco.mj_forward(_model,data);forward=data.xmat[body].reshape(3,3)[:,0]
        result.append(np.rad2deg(np.arctan2(-forward[2],np.linalg.norm(forward[:2]))))
    return np.array(result)
