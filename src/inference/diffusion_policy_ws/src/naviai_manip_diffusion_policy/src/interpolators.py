#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Dict, Any, Optional

from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

class TrajectoryInterpolator:
    def __init__(self):
        self.spline = None
        self.t_min = None
        self.t_max = None
        self.t0 = None

    def update_actions(self, actions: List[Dict[str, Any]]):
        if not actions:
            return
        times_abs = np.array([a["timestamp"] for a in actions], dtype=float)
        values = np.array([a["values"] for a in actions], dtype=float)
        self.t0 = times_abs[0]
        times = times_abs - self.t0
        self.spline = CubicSpline(times, values, axis=0)
        self.t_min = times[0]
        self.t_max = times[-1]

    def get_action(self, t_query: float) -> Dict[str, Any]:
        if self.spline is None:
            return {"timestamp": t_query, "values": None, "velocity": None}
        t_rel = np.clip(t_query - self.t0, self.t_min, self.t_max)
        values = self.spline(t_rel)
        velocity = self.spline(t_rel, 1)
        return {"timestamp": t_query, "values": values, "velocity": velocity}


class QuaternionTrajectoryInterpolator:
    def __init__(self):
        self.qs = None
        self.ts = None
        self.controls = None
        self.t_min = None
        self.t_max = None
        self.t0 = None

    @staticmethod
    def mul_quat(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    @staticmethod
    def slerp(q1, q2, t: float):
        times = [0, 1]
        rots = Rotation.from_quat([q1, q2])
        slerp_obj = Slerp(times, rots)
        rot_t = slerp_obj([t])
        return rot_t.as_quat()[0]

    @staticmethod
    def log_quat(q):
        q = q / np.linalg.norm(q)
        v = q[1:]
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            return np.zeros(3)
        theta = 2 * np.arctan2(norm_v, q[0])
        return theta * v / norm_v

    @staticmethod
    def exp_quat(v):
        theta = np.linalg.norm(v)
        if theta < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = v / theta
        return np.array([np.cos(theta/2), *(axis*np.sin(theta/2))])

    @staticmethod
    def slerp(q1, q2, t):
        key_times = [0, 1]
        key_rots = Rotation.from_quat([q1, q2])
        slerp_obj = Slerp(key_times, key_rots)
        return slerp_obj([t]).as_quat()[0]

    def _compute_control_points(self):
        n = len(self.qs)
        controls = []
        for i in range(n):
            if i == 0 or i == n-1:
                controls.append(self.qs[i])
            else:
                q_prev, q, q_next = self.qs[i-1], self.qs[i], self.qs[i+1]
                q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
                log_term = 0.25 * (self.log_quat(self.mul_quat(q_inv, q_prev)) +
                                   self.log_quat(self.mul_quat(q_inv, q_next)))
                a_i = self.mul_quat(q, self.exp_quat(-log_term))
                controls.append(a_i / np.linalg.norm(a_i))
        return np.array(controls)

    def squad(self, q1, q2, a1, a2, t):
        s1 = self.slerp(q1, q2, t)
        s2 = self.slerp(a1, a2, t)
        return self.slerp(s1, s2, 2*t*(1-t))

    def update_actions(self, actions: List[Dict[str, Any]], value_slice: Optional[slice] = None):
        """
        Update the interpolator with a list of actions.

        Args:
            actions: List of dictionaries, each containing:
                - "timestamp": float
                - "values": np.ndarray (expected length 4 for quaternion)
            value_slice: Optional slice object to select a subset of each action's values.
                        If None, all values are used.
        Raises:
            ValueError: If the resulting values length is not 4.
        """
        if not actions:
            print("AAAASSS")
            return
        times_abs = np.array([a["timestamp"] for a in actions], dtype=float)
        quats = np.array([a["values"][value_slice] if value_slice else a["values"] for a in actions], dtype=float)
        if quats.shape[1] != 4:
            raise ValueError(f"Each action's 'values' must have length 4 after slicing, got {quats.shape[1]}.")
        self.t0 = times_abs[0]
        times = times_abs - self.t0
        self.qs = np.array([q / np.linalg.norm(q) for q in quats])
        self.ts = times
        self.controls = self._compute_control_points()
        self.t_min = times[0]
        self.t_max = times[-1]

    def get_action(self, t_query: float) -> Dict[str, Any]:
        if self.qs is None:
            return {"timestamp": t_query, "values": None, "velocity": None}
        t_rel = np.clip(t_query - self.t0, self.t_min, self.t_max)
        i = np.searchsorted(self.ts, t_rel) - 1
        if i < 0:
            delta_rot = Rotation.from_quat(self.qs[1]) * Rotation.from_quat(self.qs[0]).inv()
            velocity = delta_rot.as_rotvec() / (self.ts[1] - self.ts[0])
            return {"timestamp": t_query, "values": self.qs[0], "velocity": velocity}
        if i >= len(self.qs) - 1:
            delta_rot = Rotation.from_quat(self.qs[-1]) * Rotation.from_quat(self.qs[-2]).inv()
            velocity = delta_rot.as_rotvec() / (self.ts[-1] - self.ts[-2])
            return {"timestamp": t_query, "values": self.qs[-1], "velocity": velocity}
        t0, t1 = self.ts[i], self.ts[i+1]
        local_t = (t_rel - t0) / (t1 - t0)
        q1, q2 = self.qs[i], self.qs[i+1]
        a1, a2 = self.controls[i], self.controls[i+1]
        q_interp = self.squad(q1, q2, a1, a2, local_t)
        eps = 1e-3
        q_plus = self.squad(q1, q2, a1, a2, min(local_t+eps, 1.0))
        dq = self.mul_quat(q_plus, [q_interp[0], -q_interp[1], -q_interp[2], -q_interp[3]])
        angular_velocity = self.log_quat(dq) / ((t1 - t0) * eps)
        return {"timestamp": t_query, "values": q_interp, "velocity": angular_velocity}
