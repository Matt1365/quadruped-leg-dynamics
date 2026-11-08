"""Headless Drake simulation of the leg with a pluggable torque controller."""
import numpy as np
from pydrake.geometry import HalfSpace, MeshcatVisualizer
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder


class LegSimulation:
    """Discrete-time MultibodyPlant of the URDF with a ground plane.

    controller(t, q, v) -> tau is evaluated once per time step and held
    constant during the step (zero-order hold).
    """

    def __init__(self, urdf_path, q0, dt=1e-3, meshcat=None):
        self.dt = dt
        builder = DiagramBuilder()
        self.plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=dt)
        Parser(self.plant).AddModels(urdf_path)
        self._add_ground()
        self.plant.Finalize()
        if meshcat is not None:
            MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        self.diagram = builder.Build()
        self.simulator = Simulator(self.diagram)
        self.context = self.plant.GetMyMutableContextFromRoot(self.simulator.get_mutable_context())
        self.plant.SetPositions(self.context, np.asarray(q0, dtype=float))
        self.simulator.Initialize()

    def _add_ground(self):
        friction = CoulombFriction(static_friction=0.7, dynamic_friction=0.5)
        world = self.plant.world_body()
        self.plant.RegisterCollisionGeometry(world, RigidTransform(), HalfSpace(), "ground", friction)

    def run(self, controller, duration, realtime_rate=0.0):
        """Simulate for `duration` seconds; returns time T, positions Q, velocities V, torques U."""
        self.simulator.set_target_realtime_rate(realtime_rate)
        actuation = self.plant.get_actuation_input_port()
        t0 = self.simulator.get_context().get_time()
        n = int(round(duration / self.dt))
        T, Q, V, U = (np.zeros(n + 1), np.zeros((n + 1, 3)), np.zeros((n + 1, 3)), np.zeros((n + 1, 3)))
        for k in range(n + 1):
            t = t0 + k * self.dt
            q, v = self.plant.GetPositions(self.context), self.plant.GetVelocities(self.context)
            tau = np.asarray(controller(t - t0, q, v), dtype=float)
            T[k], Q[k], V[k], U[k] = t - t0, q, v, tau
            if k < n:
                actuation.FixValue(self.context, tau)
                self.simulator.AdvanceTo(t + self.dt)
        return T, Q, V, U
