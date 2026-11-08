"""Consistency tests: closed-form kinematics and Lagrangian dynamics vs. Pinocchio."""
import numpy as np
import pinocchio as pin
import pytest

from leg_model import URDF_PATH, FOOT_FRAME
from leg_model.kinematics import forward_kinematics, joint_positions
from leg_model import planar_dynamics as pdm


@pytest.fixture(scope="module")
def robot():
    model = pin.buildModelFromUrdf(URDF_PATH)
    return model, model.createData()


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_forward_kinematics_matches_pinocchio(robot, rng):
    model, data = robot
    foot = model.getFrameId(FOOT_FRAME)
    for _ in range(200):
        q = rng.uniform(-np.pi, np.pi, 3)
        pin.framesForwardKinematics(model, data, q)
        assert np.allclose(forward_kinematics(q), data.oMf[foot].translation, atol=1e-12)


def test_knee_position_matches_pinocchio(robot, rng):
    model, data = robot
    calf = model.getJointId("FR_calf_joint")
    for _ in range(50):
        q = rng.uniform(-np.pi, np.pi, 3)
        pin.forwardKinematics(model, data, q)
        assert np.allclose(joint_positions(q)[3], data.oMi[calf].translation, atol=1e-12)


def test_jacobian_matches_finite_differences(robot, rng):
    model, data = robot
    foot = model.getFrameId(FOOT_FRAME)
    eps = 1e-6
    for _ in range(50):
        q = rng.uniform(-np.pi, np.pi, 3)
        J = pin.computeFrameJacobian(model, data, q, foot, pin.LOCAL_WORLD_ALIGNED)
        J_fd = np.column_stack([(forward_kinematics(q + eps * e) - forward_kinematics(q - eps * e)) / (2 * eps)
                                for e in np.eye(3)])
        assert np.allclose(J[:3], J_fd, atol=1e-8)


def test_lagrangian_dynamics_match_pinocchio(robot, rng):
    model, data = robot
    M, C, G = pdm.make_numeric(pdm.params_from_pinocchio(model))
    for _ in range(100):
        qs, vs = rng.uniform(-np.pi, np.pi, 2), rng.uniform(-5, 5, 2)
        q, v = np.r_[0.0, qs], np.r_[0.0, vs]
        M_pin = pin.crba(model, data, q)
        M_pin = np.triu(M_pin) + np.triu(M_pin, 1).T
        G_pin = pin.computeGeneralizedGravity(model, data, q)
        C_pin = pin.nonLinearEffects(model, data, q, v) - G_pin
        assert np.allclose(M_pin[1:, 1:], M(qs), atol=1e-12)
        assert np.allclose(G_pin[1:], G(qs), atol=1e-12)
        assert np.allclose(C_pin[1:], C(qs, vs), atol=1e-12)


def test_gravity_compensation_holds_pose_in_drake(robot):
    pytest.importorskip("pydrake")
    from leg_model.simulation import LegSimulation
    model, data = robot
    q0 = np.array([0.0, 0.44, -1.33])
    _, Q, _, _ = LegSimulation(URDF_PATH, q0).run(
        lambda t, q, v: pin.computeGeneralizedGravity(model, data, q), duration=1.0)
    assert np.abs(Q - q0).max() < 1e-8
