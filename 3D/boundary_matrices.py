import numpy as np
from fem_types import Element, GlobalData, Grid
from config import NUMBER_OF_INTEGRATION_POINTS
from gauss_integration import GAUSS_QUADRATURE

HEX_FACES = [
    [0, 3, 2, 1],  # -Z
    [4, 5, 6, 7],  # +Z
    [0, 1, 5, 4],  # -Y
    [1, 2, 6, 5],  # +X
    [2, 3, 7, 6],  # +Y
    [3, 0, 4, 7],  # -X
]


def get_face_shape_fns_and_derivs(
    u: float, v: float
) -> tuple[np.array, np.array, np.array]:
    N = np.array(
        [
            0.25 * (1 - u) * (1 - v),
            0.25 * (1 + u) * (1 - v),
            0.25 * (1 + u) * (1 + v),
            0.25 * (1 - u) * (1 + v),
        ]
    )
    dN_du = np.array([-0.25 * (1 - v), 0.25 * (1 - v), 0.25 * (1 + v), -0.25 * (1 + v)])
    dN_dv = np.array([-0.25 * (1 - u), -0.25 * (1 + u), 0.25 * (1 + u), 0.25 * (1 - u)])
    return N, dN_du, dN_dv


def generate_Hbc_matrix_and_P_vector(
    element: Element, globalData: GlobalData, grid: Grid
) -> tuple[np.matrix, np.matrix]:

    Hbc_matrix = np.zeros((8, 8))
    P_vector = np.zeros(8)

    gauss_points = GAUSS_QUADRATURE[NUMBER_OF_INTEGRATION_POINTS]["nodes"]
    weights = GAUSS_QUADRATURE[NUMBER_OF_INTEGRATION_POINTS]["weights"]

    for face_local_ids in HEX_FACES:
        is_boundary_face = True
        for local_id in face_local_ids:
            global_id = element.node_ids[local_id]
            if not grid.nodes[global_id - 1].convection_bc:
                is_boundary_face = False
                break

        if not is_boundary_face:
            continue

        face_nodes = [grid.nodes[element.node_ids[lid] - 1] for lid in face_local_ids]
        xs = np.array([n.x for n in face_nodes])
        ys = np.array([n.y for n in face_nodes])
        zs = np.array([n.z for n in face_nodes])

        for i, u in enumerate(gauss_points):
            for j, v in enumerate(gauss_points):
                weight = weights[i] * weights[j]
                N_2d, dN_du, dN_dv = get_face_shape_fns_and_derivs(u, v)

                dx_du = np.sum(dN_du * xs)
                dy_du = np.sum(dN_du * ys)
                dz_du = np.sum(dN_du * zs)
                t_u = np.array([dx_du, dy_du, dz_du])

                dx_dv = np.sum(dN_dv * xs)
                dy_dv = np.sum(dN_dv * ys)
                dz_dv = np.sum(dN_dv * zs)
                t_v = np.array([dx_dv, dy_dv, dz_dv])

                normal_vector = np.cross(t_u, t_v)
                detJ_surf = np.linalg.norm(normal_vector)

                N_3d_on_face = np.zeros(8)
                for k, local_node_idx in enumerate(face_local_ids):
                    N_3d_on_face[local_node_idx] = N_2d[k]
                Hbc_matrix += (
                    np.outer(N_3d_on_face, N_3d_on_face)
                    * globalData.Alpha
                    * detJ_surf
                    * weight
                )
                P_vector += (
                    N_3d_on_face
                    * globalData.Alpha
                    * globalData.Tenv
                    * detJ_surf
                    * weight
                )

    return Hbc_matrix, P_vector
