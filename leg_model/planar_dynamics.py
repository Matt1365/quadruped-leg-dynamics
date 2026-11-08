"""Euler-Lagrange model of the thigh + calf chain in the sagittal (x-z) plane.

Valid when the hip (abduction) joint is held at zero, so thigh and calf only
rotate about the y-axis. Uses the same sign convention as the URDF:
a positive thigh angle swings the foot towards -x, and the calf angle is
measured relative to the thigh.

Each link is a rigid body with mass m_i, centre of mass at r_i from its joint
and rotational inertia I_i about its centre of mass (y-axis). An optional point
mass m_f sits at the end of the calf (the foot).
"""
import numpy as np
import sympy as sp

q2, q3 = sp.symbols("q2 q3")              # thigh, calf (relative)
dq2, dq3 = sp.symbols("dq2 dq3")
ddq2, ddq3 = sp.symbols("ddq2 ddq3")
m1, m2, mf = sp.symbols("m1 m2 m_f", nonnegative=True)
l1, l2, r1, r2 = sp.symbols("l1 l2 r1 r2", positive=True)
I1, I2 = sp.symbols("I1 I2", nonnegative=True)
g = sp.symbols("g", positive=True)

q = sp.Matrix([q2, q3])
dq = sp.Matrix([dq2, dq3])
ddq = sp.Matrix([ddq2, ddq3])


def _point(dist_thigh, dist_calf):
    # Position (x, z) relative to the thigh joint
    x = -dist_thigh * sp.sin(q2) - dist_calf * sp.sin(q2 + q3)
    z = -dist_thigh * sp.cos(q2) - dist_calf * sp.cos(q2 + q3)
    return sp.Matrix([x, z])


p_c1 = _point(r1, 0)       # thigh centre of mass
p_c2 = _point(l1, r2)      # calf centre of mass
p_f = _point(l1, l2)       # foot point mass

v_c1, v_c2, v_f = (p.jacobian(q) * dq for p in (p_c1, p_c2, p_f))

# Kinetic energy: translation of each mass + rotation of each link
T = (sp.Rational(1, 2) * m1 * v_c1.dot(v_c1) + sp.Rational(1, 2) * I1 * dq2**2
     + sp.Rational(1, 2) * m2 * v_c2.dot(v_c2) + sp.Rational(1, 2) * I2 * (dq2 + dq3)**2
     + sp.Rational(1, 2) * mf * v_f.dot(v_f))
V = g * (m1 * p_c1[1] + m2 * p_c2[1] + mf * p_f[1])
L = T - V

# d/dt(dL/ddq) - dL/dq, expanded with the chain rule (no explicit time dependence)
EL = sp.Matrix([
    sum(sp.diff(L, dq[i], q[j]) * dq[j] + sp.diff(L, dq[i], dq[j]) * ddq[j] for j in range(2))
    - sp.diff(L, q[i])
    for i in range(2)
])

M_sym = sp.simplify(EL.jacobian(ddq))
G_sym = sp.simplify(sp.Matrix([V]).jacobian(q).T)
C_sym = sp.simplify(EL.subs({ddq2: 0, ddq3: 0}) - G_sym)

PARAM_SYMBOLS = (m1, m2, mf, l1, l2, r1, r2, I1, I2, g)


def make_numeric(params):
    """Return numpy functions M(q), C(q, dq), G(q) for a dict of parameter values."""
    subs = {s: params[s.name] for s in PARAM_SYMBOLS}
    M_fn = sp.lambdify((q2, q3), M_sym.subs(subs), "numpy")
    C_fn = sp.lambdify((q2, q3, dq2, dq3), C_sym.subs(subs), "numpy")
    G_fn = sp.lambdify((q2, q3), G_sym.subs(subs), "numpy")
    return (lambda qv: np.array(M_fn(*qv), dtype=float),
            lambda qv, dqv: np.array(C_fn(*qv, *dqv), dtype=float).ravel(),
            lambda qv: np.array(G_fn(*qv), dtype=float).ravel())


def params_from_pinocchio(model, g_value=9.81):
    """Read link masses, CoM offsets and y-inertias from a Pinocchio model of the URDF."""
    def inertia(joint_name):
        return model.inertias[model.getJointId(joint_name)]
    thigh, calf = inertia("FR_thigh_joint"), inertia("FR_calf_joint")
    # The fixed foot frame is merged into the calf body by Pinocchio, so read it from the URDF frame
    foot_frame = model.frames[model.getFrameId("FR_foot")]
    l_calf = -foot_frame.placement.translation[2]
    l_thigh = -model.jointPlacements[model.getJointId("FR_calf_joint")].translation[2]
    return {
        "m1": thigh.mass, "r1": -thigh.lever[2], "I1": thigh.inertia[1, 1], "l1": l_thigh,
        "m2": calf.mass, "r2": -calf.lever[2], "I2": calf.inertia[1, 1], "l2": l_calf,
        "m_f": 0.0, "g": g_value,
    }
