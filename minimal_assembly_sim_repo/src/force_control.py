import numpy as np


class SimpleForceController:
    def __init__(self):
        self.ref_force = np.array([0.0, 0.0, -70.0, 0.0, 0.0, 0.0], dtype=float)
        self.Kpz = 0.015
        self.Krxyz = 0.01
        self.Kpxy = 0.0022

    def expert_action(self, wrench):
        e = self.ref_force - np.asarray(wrench, dtype=float)
        e[0] *= -1.0

        a = np.zeros(6, dtype=float)
        a[0:2] = self.Kpxy * e[0:2]
        a[2] = self.Kpz * e[2]
        a[3] = -self.Krxyz * e[3]
        a[4:6] = self.Krxyz * e[4:6]
        return a
