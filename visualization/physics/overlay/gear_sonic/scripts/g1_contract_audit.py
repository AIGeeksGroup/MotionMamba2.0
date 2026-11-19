"""Fail-fast audit for the G1 body-policy and Dex3 MuJoCo asset contract."""

from __future__ import annotations

from pathlib import Path

import mujoco


BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
    return [
        mujoco.mj_id2name(model, object_type, index) or f"<unnamed:{index}>"
        for index in range(count)
    ]


def audit_g1_model(sim_env, robot_scene: str | Path) -> dict[str, object]:
    """Validate and print the exact model/policy mapping used by the simulator."""
    model = sim_env.mj_model
    joint_names = _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    actuator_names = _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    mesh_names = _names(model, mujoco.mjtObj.mjOBJ_MESH, model.nmesh)
    body_joint_ids = [int(value) for value in sim_env.body_joint_index]
    left_hand_ids = [int(value) for value in sim_env.left_hand_index]
    right_hand_ids = [int(value) for value in sim_env.right_hand_index]
    controlled_names = [joint_names[index] for index in body_joint_ids]
    left_hand_names = [joint_names[index] for index in left_hand_ids]
    right_hand_names = [joint_names[index] for index in right_hand_ids]

    if tuple(controlled_names) != BODY_JOINT_NAMES:
        raise RuntimeError(
            "MuJoCo controlled-joint order does not match the 29-DoF SONIC body "
            f"contract. expected={BODY_JOINT_NAMES}, actual={tuple(controlled_names)}"
        )
    if len(set(controlled_names)) != len(controlled_names):
        raise RuntimeError("MuJoCo controlled-joint names are not unique.")
    if len(left_hand_names) != 7 or len(right_hand_names) != 7:
        raise RuntimeError(
            "The configured scene must contain seven Dex3 joints per hand; "
            f"got left={len(left_hand_names)}, right={len(right_hand_names)}."
        )
    if model.nu != 43 or model.nv != 49 or model.nq != 50:
        raise RuntimeError(
            "Unexpected G1 43-DoF scene dimensions; expected nq/nv/nu=50/49/43, "
            f"got {model.nq}/{model.nv}/{model.nu}."
        )
    if int(sim_env.robot.NUM_JOINTS) != 29:
        raise RuntimeError(
            f"SONIC body control must be 29-DoF, got {sim_env.robot.NUM_JOINTS}."
        )
    if int(sim_env.unitree_bridge.num_body_motor) != 29:
        raise RuntimeError(
            "Unitree bridge body command dimension must be 29, got "
            f"{sim_env.unitree_bridge.num_body_motor}."
        )

    body_actuator_ids: list[int] = []
    for joint_id, joint_name in zip(body_joint_ids, controlled_names, strict=True):
        matches = [
            actuator_id
            for actuator_id in range(model.nu)
            if int(model.actuator_trnid[actuator_id, 0]) == joint_id
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Joint {joint_name!r} must map to exactly one actuator, got {matches}."
            )
        body_actuator_ids.append(matches[0])
    expected_torque_slots = [joint_id - 1 for joint_id in body_joint_ids]
    if body_actuator_ids != expected_torque_slots:
        raise RuntimeError(
            "MuJoCo body torque slots do not match joint-target actuators; "
            f"expected={expected_torque_slots}, actual={body_actuator_ids}."
        )

    print("[G1 asset audit] Loaded robot asset:", Path(robot_scene).resolve(), flush=True)
    print(
        "[G1 asset audit] variant=G1 29-DoF body + Dex3-1 7-DoF per hand; "
        f"nq={model.nq} nv={model.nv} nu={model.nu} njnt={model.njnt}",
        flush=True,
    )
    print(
        "[G1 asset audit] policy_action_dim=29 controlled_joint_count=29 "
        "hand_control=separate Dex3 channel units=rad/rad_s root=Z-up/+X-forward/wxyz",
        flush=True,
    )
    for index, (joint_id, joint_name, actuator_id) in enumerate(
        zip(body_joint_ids, controlled_names, body_actuator_ids, strict=True)
    ):
        print(
            "[G1 body mapping] "
            f"{index:02d} joint_id={joint_id:02d} joint={joint_name} "
            f"qpos={int(model.jnt_qposadr[joint_id])} "
            f"qvel={int(model.jnt_dofadr[joint_id])} actuator_id={actuator_id:02d} "
            f"actuator={actuator_names[actuator_id]}",
            flush=True,
        )
    print("[G1 joint names] " + ", ".join(joint_names), flush=True)
    print("[G1 actuator names] " + ", ".join(actuator_names), flush=True)
    print("[G1 Dex3 left joints] " + ", ".join(left_hand_names), flush=True)
    print("[G1 Dex3 right joints] " + ", ".join(right_hand_names), flush=True)
    hand_mesh_names = [
        name
        for name in mesh_names
        if any(token in name for token in ("hand", "thumb", "middle", "index"))
    ]
    print("[G1 Dex3 mesh names] " + ", ".join(hand_mesh_names), flush=True)
    return {
        "asset": str(Path(robot_scene).resolve()),
        "variant": "g1_29dof_body_plus_dex3_14dof",
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "controlled_joint_names": controlled_names,
        "body_actuator_ids": body_actuator_ids,
        "left_hand_joint_names": left_hand_names,
        "right_hand_joint_names": right_hand_names,
        "hand_mesh_names": hand_mesh_names,
    }
