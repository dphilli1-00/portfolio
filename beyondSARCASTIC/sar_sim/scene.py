"""
Scene geometry: a flat ground patch + a box (building), as triangle meshes.

No Blender/Unity/Unreal -- geometry is authored directly as numpy arrays.
Everything downstream (raytrace.py) only needs: vertices, per-triangle
centroid/normal/area, and a material tag per triangle.
"""

import numpy as np


class TriMesh:
    """Flat container for a batch of triangles. All arrays are [T, ...]."""

    def __init__(self, v0, v1, v2, material_ids):
        self.v0 = np.asarray(v0, dtype=np.float64)
        self.v1 = np.asarray(v1, dtype=np.float64)
        self.v2 = np.asarray(v2, dtype=np.float64)
        self.material_ids = np.asarray(material_ids, dtype=object)

        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        cross = np.cross(edge1, edge2)
        cross_norm = np.linalg.norm(cross, axis=1)

        self.normal = cross / cross_norm[:, None]
        self.area = 0.5 * cross_norm
        self.centroid = (self.v0 + self.v1 + self.v2) / 3.0

    def __len__(self):
        return len(self.v0)

    @staticmethod
    def concat(meshes):
        return TriMesh(
            v0=np.concatenate([m.v0 for m in meshes], axis=0),
            v1=np.concatenate([m.v1 for m in meshes], axis=0),
            v2=np.concatenate([m.v2 for m in meshes], axis=0),
            material_ids=np.concatenate([m.material_ids for m in meshes], axis=0),
        )


def make_ground(size=40.0, n_subdiv=14, material="dry_soil"):
    """Flat square patch centered at origin, z=0, subdivided into 2*n^2 triangles."""
    xs = np.linspace(-size / 2, size / 2, n_subdiv + 1)
    ys = np.linspace(-size / 2, size / 2, n_subdiv + 1)

    v0_list, v1_list, v2_list = [], [], []
    for i in range(n_subdiv):
        for j in range(n_subdiv):
            p00 = np.array([xs[i], ys[j], 0.0])
            p10 = np.array([xs[i + 1], ys[j], 0.0])
            p01 = np.array([xs[i], ys[j + 1], 0.0])
            p11 = np.array([xs[i + 1], ys[j + 1], 0.0])
            # two triangles per quad, CCW when viewed from +z so normal = +z
            v0_list += [p00, p10]
            v1_list += [p10, p11]
            v2_list += [p11, p01]

    n_tri = len(v0_list)
    return TriMesh(v0_list, v1_list, v2_list, [material] * n_tri)


def make_box(center_xy, width, depth, height, wall_material="concrete", roof_material="metal"):
    """
    Axis-aligned box sitting on the ground (z=0 base), 12 triangles (2/face x 6).
    center_xy: (x, y) of the box footprint center.
    """
    cx, cy = center_xy
    x0, x1 = cx - width / 2, cx + width / 2
    y0, y1 = cy - depth / 2, cy + depth / 2
    z0, z1 = 0.0, height

    # 8 corners
    c = {
        "000": np.array([x0, y0, z0]), "100": np.array([x1, y0, z0]),
        "010": np.array([x0, y1, z0]), "110": np.array([x1, y1, z0]),
        "001": np.array([x0, y0, z1]), "101": np.array([x1, y0, z1]),
        "011": np.array([x0, y1, z1]), "111": np.array([x1, y1, z1]),
    }

    faces = []  # (v0, v1, v2, v3) wound CCW as seen from outside the box, material
    # -y wall (facing negative y), roof, and remaining walls
    faces.append((c["000"], c["100"], c["101"], c["001"], wall_material))  # -y wall
    faces.append((c["110"], c["010"], c["011"], c["111"], wall_material))  # +y wall
    faces.append((c["010"], c["000"], c["001"], c["011"], wall_material))  # -x wall
    faces.append((c["100"], c["110"], c["111"], c["101"], wall_material))  # +x wall
    faces.append((c["001"], c["101"], c["111"], c["011"], roof_material))  # roof (+z)
    # base is on the ground, not radar-visible -- skip it, saves triangles

    v0_list, v1_list, v2_list, mats = [], [], [], []
    for a, b, cc, d, mat in faces:
        v0_list += [a, a]
        v1_list += [b, cc]
        v2_list += [cc, d]
        mats += [mat, mat]

    return TriMesh(v0_list, v1_list, v2_list, mats)


def build_scene(box_center=(8.0, 3.0), box_size=(6.0, 4.0, 5.0)):
    ground = make_ground()
    box = make_box(box_center, *box_size)
    return TriMesh.concat([ground, box])
