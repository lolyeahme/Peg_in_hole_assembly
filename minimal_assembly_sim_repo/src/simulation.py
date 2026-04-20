import numpy as np


class DualPegInHoleSimulation:
    def __init__(self, init_transform):
        self.diameter_hole = 30.0
        self.diameter_peg = 29.9
        self.distance_axes = 200.0
        self.height = 100.0
        self.K = 100.0
        self.u = 0.05
        self.num_points = 4000

        self.centers_pegs, self.centers_holes, self.pegs, self.holes = self._create_model()
        self.apply_transform(init_transform)

    def _circle(self, center, radius):
        theta = np.linspace(0.0, 2.0 * np.pi, self.num_points, endpoint=False)
        out = np.zeros((3, self.num_points), dtype=float)
        out[0, :] = radius * np.cos(theta) + center[0]
        out[1, :] = radius * np.sin(theta) + center[1]
        out[2, :] = center[2]
        return out

    def _create_model(self):
        d = self.distance_axes / 2.0
        h = self.height
        centers_pegs = np.array([
            [-d, d, -d, d],
            [0.0, 0.0, 0.0, 0.0],
            [2 * h + 1, 2 * h + 1, h + 1, h + 1],
        ], dtype=float)
        centers_holes = np.array([
            [-d, d, -d, d],
            [0.0, 0.0, 0.0, 0.0],
            [h, h, 0.0, 0.0],
        ], dtype=float)
        pegs = np.zeros((4, 3, self.num_points), dtype=float)
        holes = np.zeros((4, 3, self.num_points), dtype=float)
        for i in range(4):
            pegs[i] = self._circle(centers_pegs[:, i], self.diameter_peg / 2.0)
            holes[i] = self._circle(centers_holes[:, i], self.diameter_hole / 2.0)
        return centers_pegs, centers_holes, pegs, holes

    @staticmethod
    def _rot_xyz_deg(rx, ry, rz):
        rx, ry, rz = np.deg2rad([rx, ry, rz])
        rotx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=float)
        roty = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=float)
        rotz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=float)
        return rotz @ roty @ rotx

    def _cal_peg_in_hole(self, centers):
        z_axis = centers[:, 2] - centers[:, 0]
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = centers[:, 1] - centers[:, 0]
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        T = np.eye(4, dtype=float)
        T[0:3, 0] = x_axis
        T[0:3, 1] = y_axis
        T[0:3, 2] = z_axis
        T[0:3, 3] = (centers[:, 0] + centers[:, 1]) / 2.0
        return T

    def apply_transform(self, transform):
        rx, ry, rz, dx, dy, dz = transform
        peg_in_hole = self._cal_peg_in_hole(self.centers_pegs)
        hole_in_peg = np.linalg.inv(peg_in_hole)

        rotation_in_hole = peg_in_hole[0:3, 0:3] @ self._rot_xyz_deg(rx, ry, rz)
        translation_in_hole = peg_in_hole[0:3, 0:3] @ np.array([dx, dy, dz], dtype=float) + peg_in_hole[0:3, 3]

        centers_local = hole_in_peg[0:3, 0:3] @ self.centers_pegs + hole_in_peg[0:3, 3:4]
        self.centers_pegs = rotation_in_hole @ centers_local + translation_in_hole[:, None]

        pegs_new = np.zeros_like(self.pegs)
        for i in range(4):
            peg_local = hole_in_peg[0:3, 0:3] @ self.pegs[i] + hole_in_peg[0:3, 3:4]
            pegs_new[i] = rotation_in_hole @ peg_local + translation_in_hole[:, None]
        self.pegs = pegs_new

        self.peg_in_hole = self._cal_peg_in_hole(self.centers_pegs)
        self.hole_in_peg = np.linalg.inv(self.peg_in_hole)
        self.position = self._pose6_from_transform(self.peg_in_hole)
        self.force = self._sum_contact_force()
        return self.force.copy(), self.position.copy()

    @staticmethod
    def _pose6_from_transform(T):
        rz = np.arctan2(T[1, 0], T[0, 0])
        ry = np.arctan2(-T[2, 0], np.cos(rz) * T[0, 0] + np.sin(rz) * T[1, 0])
        rx = np.arctan2(np.sin(rz) * T[0, 2] - np.cos(rz) * T[1, 2], -np.sin(rz) * T[0, 1] + np.cos(rz) * T[1, 1])
        return np.array([T[0, 3], T[1, 3], T[2, 3], np.rad2deg(rx), np.rad2deg(ry), np.rad2deg(rz)], dtype=float)

    @staticmethod
    def _find_down_points(center_hole_xy, r_hole, circle_peg):
        pts = circle_peg.T
        dxy = pts[:, 0:2] - center_hole_xy[None, :]
        dist = np.linalg.norm(dxy, axis=1)
        pen = np.maximum(dist - r_hole, 0.0)
        m = pen.max()
        ids = np.where((pen == m) & (pen > 0.0))[0]
        out = np.zeros((len(ids), 7), dtype=float)
        for i, idx in enumerate(ids):
            out[i, 0:3] = pts[idx, 0:3]
            out[i, 3] = pen[idx]
            out[i, 4] = 0.0
            out[i, 5:7] = center_hole_xy - pts[idx, 0:2]
        return out

    @staticmethod
    def _find_up_points(peg_center_xy, r_peg, circle_hole):
        pts = circle_hole.T
        dxy = pts[:, 0:2] - peg_center_xy[None, :]
        dist = np.linalg.norm(dxy, axis=1)
        pen = np.maximum(r_peg - dist, 0.0)
        m = dist.min()
        ids = np.where((dist == m) & (dist < r_peg))[0]
        out = np.zeros((len(ids), 7), dtype=float)
        for i, idx in enumerate(ids):
            out[i, 0:3] = pts[idx, 0:3]
            out[i, 3] = pen[idx]
            out[i, 4] = 0.0
            out[i, 5:7] = peg_center_xy - pts[idx, 0:2]
        return out

    @staticmethod
    def _point_force(point, K, u):
        fn_xy = K * point[3]
        norm = np.hypot(point[5], point[6]) + 1e-12
        fx = fn_xy * point[5] / norm
        fy = fn_xy * point[6] / norm
        fz = fn_xy * u + K * point[4]
        return np.array([fx, fy, fz], dtype=float)

    def _down_contact_force(self, contour, hole_center, hole_radius):
        pts = self._find_down_points(hole_center[0:2], hole_radius, contour)
        out = np.zeros(6, dtype=float)
        for p in pts:
            f_h = self._point_force(p, self.K, self.u)
            c_h = np.hstack([p[0:3], 1.0])
            c_p = self.hole_in_peg @ c_h
            f_p = self.hole_in_peg[0:3, 0:3] @ f_h
            out[0:3] += f_p
            out[3:6] += np.cross(c_p[0:3] / 1000.0, f_p)
        return out

    def _up_contact_force(self, hole_circle, peg_center, peg_radius):
        peg_center_p = self.hole_in_peg[0:3, 0:3] @ peg_center + self.hole_in_peg[0:3, 3]
        hole_circle_p = self.hole_in_peg[0:3, 0:3] @ hole_circle + self.hole_in_peg[0:3, 3:4]
        pts = self._find_up_points(peg_center_p[0:2], peg_radius, hole_circle_p)
        out = np.zeros(6, dtype=float)
        for p in pts:
            f_p = self._point_force(p, self.K, self.u)
            f_p[2] *= -1.0
            out[0:3] += f_p
            out[3:6] += np.cross(p[0:3] / 1000.0, f_p)
        return out

    def _sum_contact_force(self):
        total = np.zeros(6, dtype=float)
        hole_r = self.diameter_hole / 2.0
        peg_r = self.diameter_peg / 2.0

        for i in range(2):
            fd = self._down_contact_force(self.pegs[i + 2], self.centers_holes[:, i + 2], hole_r)
            fu = self._up_contact_force(self.holes[i], self.centers_pegs[:, i], peg_r)
            if fd[0] * fu[0] <= 0 and fd[1] * fu[1] <= 0:
                total += fd + fu
            else:
                total += fd

        noise_scale = 0.1 * np.abs(total)
        total = np.random.normal(total, noise_scale)
        total[2] = np.clip(total[2], -100.0, 0.0)
        return total
