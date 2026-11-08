"""Kinematics and dynamics model of a quadruped front-right leg."""
from pathlib import Path

URDF_PATH = str(Path(__file__).resolve().parent.parent / "urdf" / "front_right_leg.urdf")
FOOT_FRAME = "FR_foot"
