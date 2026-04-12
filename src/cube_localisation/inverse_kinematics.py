"""Inverse kinematics solver used by cube-localisation data generation."""

from __future__ import annotations

from math import acos, asin, atan, atan2, degrees, radians, sqrt

import numpy as np

from cube_localisation.forward_kinematics import DHForwardKinematics

try:
    from scipy.spatial.transform import Rotation as _Rotation
except ImportError:
    _Rotation = None


class InverseKinematics:
    """
    6-DOF inverse-kinematics solver for the Robot V2 geometry.

    Ported from Lyra backend `src/backend/kinematics/inverse_kinematics.py`.
    """

    def __init__(self) -> None:
        # End-effector properties
        self.end_eff_pos: list[float] | None = None
        self.end_eff_rot: list[float] | None = None

        # Arm properties
        self.pri_arm_origin = [[0.0], [0.0], [163.2]]
        self.pri_arm_length = [[0.0], [0.0], [236.5]]
        self.sec_arm_length = [[0.0], [97.5], [236.5]]
        self.ter_arm_length = [[0.0], [-255.0], [0.0]]

        # Joint angles (degrees)
        self.j0 = 0.0
        self.j1 = 0.0
        self.j2 = 0.0
        self.j3 = 0.0
        self.j4 = 0.0
        self.j5 = 0.0

        theta = [radians(90.0), radians(90.0), radians(0.0), radians(-90.0)]
        alpha = [radians(-90.0), radians(0.0), radians(-90.0), radians(180.0)]
        radius = [0.0, -self.sec_arm_length[2][0], -self.sec_arm_length[1][0], 0.0]
        distance = [self.pri_arm_origin[0][0], 0.0, 0.0, self.get_arm_length(self.sec_arm_length)]

        self.fk = DHForwardKinematics(theta, alpha, radius, distance)

    def set_end_effector(self, position: list[float], rotation: list[float]) -> None:
        """Update desired end-effector position (mm) and rotation (deg)."""
        self.end_eff_pos = [float(value) for value in position]
        self.end_eff_rot = [float(value) for value in rotation]

    def calc_inverse_kinematics(self) -> list[float]:
        """Calculate actuator angles in degrees for the configured end-effector pose."""
        if self.end_eff_pos is None or self.end_eff_rot is None:
            raise ValueError("End-effector position and rotation must be set before solving IK.")
        if _Rotation is None:
            raise RuntimeError(
                "scipy is required for pregrab IK mode. Install with: pip install scipy"
            )

        end_eff_rot = _Rotation.from_euler(
            "xyz",
            [radians(self.end_eff_rot[0]), radians(self.end_eff_rot[1]), radians(self.end_eff_rot[2])],
        )
        end_eff_mat = end_eff_rot.as_matrix()

        # Target position for joints j0/j1/j2 (same approach as Lyra backend).
        target_3_mat = end_eff_mat @ self.ter_arm_length
        target_3_x = float(target_3_mat.item(0)) + self.end_eff_pos[0]
        target_3_y = float(target_3_mat.item(1)) + self.end_eff_pos[1]
        target_3_z = float(target_3_mat.item(2)) + self.end_eff_pos[2]

        self.j0 = degrees(atan2(target_3_x, target_3_y))

        hyp = sqrt(target_3_x**2 + target_3_y**2)
        dz = self.pri_arm_origin[2][0] - target_3_z

        d_hyp = sqrt(dz**2 + hyp**2)
        d_hyp_rot = degrees(asin(dz / d_hyp))

        pri_arm_length = self.get_arm_length(self.pri_arm_length)
        sec_arm_length = self.get_arm_length(self.sec_arm_length)

        if pri_arm_length + sec_arm_length < d_hyp:
            d_hyp = pri_arm_length + sec_arm_length

        cos_sentence = degrees(
            acos((sec_arm_length**2 - pri_arm_length**2 - d_hyp**2) / (-2.0 * d_hyp * pri_arm_length))
        )
        self.j1 = 90.0 - (cos_sentence - d_hyp_rot)

        self.j2 = degrees(
            acos((d_hyp**2 - sec_arm_length**2 - pri_arm_length**2) / (-2.0 * sec_arm_length * pri_arm_length))
        )
        self.j2 -= degrees(atan(self.sec_arm_length[1][0] / self.sec_arm_length[2][0]))

        self.fk.set_joint_angles(180.0 - self.j0, -self.j1, -(90.0 - self.j2), 0.0)
        joint3_rot = self.fk.get_joint_rotation_matrix(3)
        rel_rot = np.matmul(np.linalg.inv(joint3_rot), end_eff_mat)

        rel_rot_obj = _Rotation.from_matrix(rel_rot)
        angles_yxz = rel_rot_obj.as_euler("yxz", degrees=True)

        raw_j5, raw_j4, raw_j3 = angles_yxz
        self.j4 = -raw_j4 if -90.0 <= raw_j3 <= 90.0 else 180.0 + raw_j4
        self.j3 = (raw_j3 + 90.0) % 180.0 - 90.0
        self.j5 = ((-raw_j5) + 90.0) % 180.0 - 90.0

        return [self.j0, self.j1, self.j2, self.j3, self.j4, self.j5]

    @staticmethod
    def get_arm_length(arm: list[list[float]]) -> float:
        return sqrt(sum(value[0] ** 2 for value in arm))
