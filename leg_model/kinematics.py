"""Closed-form forward kinematics of the front-right leg.

Joint order q = [hip, thigh, calf]:
  - hip   rotates about the base x-axis (abduction / adduction)
  - thigh rotates about the y-axis      (flexion / extension)
  - calf  rotates about the y-axis      (knee flexion / extension)
Zero configuration: leg fully stretched along the negative base z-axis.
"""
import numpy as np

# Geometry (metres), identical to urdf/front_right_leg.urdf
HIP_X = 0.1934        # hip joint position in the body frame
HIP_Y = -0.0465
HIP_OFFSET = 0.0955   # lateral offset from hip to thigh joint
THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213
BASE_HEIGHT = 0.5     # body is welded 0.5 m above the world origin (keeps the leg off the ground)


def forward_kinematics(q, base_height=BASE_HEIGHT):
    """Foot position p = [x, y, z] in the world frame for joint angles q."""
    q1, q2, q3 = q
    # Sagittal-plane reach of the thigh + calf chain (q2 + q3 is the calf's absolute pitch)
    reach_x = THIGH_LENGTH * np.sin(q2) + CALF_LENGTH * np.sin(q2 + q3)
    reach_z = THIGH_LENGTH * np.cos(q2) + CALF_LENGTH * np.cos(q2 + q3)

    x = HIP_X - reach_x
    y = HIP_Y - HIP_OFFSET * np.cos(q1) + reach_z * np.sin(q1)
    z = base_height - HIP_OFFSET * np.sin(q1) - reach_z * np.cos(q1)
    return np.array([x, y, z])


def joint_positions(q, base_height=BASE_HEIGHT):
    """World positions of body origin, hip, thigh joint, knee and foot (for plotting)."""
    q1, q2, q3 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    body = np.array([0.0, 0.0, base_height])
    hip = body + np.array([HIP_X, HIP_Y, 0.0])
    thigh = hip + np.array([0.0, -HIP_OFFSET * c1, -HIP_OFFSET * s1])
    # Unit vector "down the leg" after hip roll, scaled by sagittal components
    def chain_point(dx, dz):
        return thigh + np.array([-dx, dz * s1, -dz * c1])
    knee = chain_point(THIGH_LENGTH * np.sin(q2), THIGH_LENGTH * np.cos(q2))
    foot = forward_kinematics(q, base_height)
    return np.vstack([body, hip, thigh, knee, foot])
