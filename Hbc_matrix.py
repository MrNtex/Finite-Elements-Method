from typing import List
import numpy as np
from math import sqrt

from fem_types import Element, GlobalData
from config import NUMBER_OF_INTEGRATION_POINTS
from gauss_integration import GAUSS_QUADRATURE
    
def generate_shape_functions_at_boundary() -> np.matrix:
    shape_functions = np.zeros((4, NUMBER_OF_INTEGRATION_POINTS, 4))

    for i, xi in enumerate(GAUSS_QUADRATURE[NUMBER_OF_INTEGRATION_POINTS]["nodes"]):
        # Bottom edge (eta = -1)
        shape_functions[0, i, 0] = 0.25 * (1 - xi) * (1 - (-1))
        shape_functions[0, i, 1] = 0.25 * (1 + xi) * (1 - (-1))
        shape_functions[0, i, 2] = 0.25 * (1 + xi) * (1 + (-1))
        shape_functions[0, i, 3] = 0.25 * (1 - xi) * (1 + (-1))

        # Right edge (ksi = 1)
        shape_functions[1, i, 0] = 0.25 * (1 - 1) * (1 - xi)
        shape_functions[1, i, 1] = 0.25 * (1 + 1) * (1 - xi)
        shape_functions[1, i, 2] = 0.25 * (1 + 1) * (1 + xi)
        shape_functions[1, i, 3] = 0.25 * (1 - 1) * (1 + xi)

        # Top edge (eta = 1)
        shape_functions[2, i, 0] = 0.25 * (1 - xi) * (1 - 1)
        shape_functions[2, i, 1] = 0.25 * (1 + xi) * (1 - 1)
        shape_functions[2, i, 2] = 0.25 * (1 + xi) * (1 + 1)
        shape_functions[2, i, 3] = 0.25 * (1 - xi) * (1 + 1)

        # Left edge (ksi = -1)
        shape_functions[3, i, 0] = 0.25 * (1 - (-1)) * (1 - xi)
        shape_functions[3, i, 1] = 0.25 * (1 + (-1)) * (1 - xi)
        shape_functions[3, i, 2] = 0.25 * (1 + (-1)) * (1 + xi)
        shape_functions[3, i, 3] = 0.25 * (1 - (-1)) * (1 + xi)
    return shape_functions

def generate_Hbc_matrix_for_side(
    element: Element,
    node_ids: tuple[int, int],
    globalData: GlobalData,
    shape_functions: np.matrix
) -> np.matrix:
    Hbc_matrix = np.zeros((4, 4))

    detJ = sqrt(element.node_ids[node_ids[0]]**2 + element.node_ids[node_ids[1]]**2) / 2

    for ip_index in range(NUMBER_OF_INTEGRATION_POINTS):
        weight = GAUSS_QUADRATURE[NUMBER_OF_INTEGRATION_POINTS]["weights"][ip_index]

        partial_Hbc_matrix = np.outer(shape_functions[ip_index], shape_functions[ip_index]) * weight * detJ * globalData.Conductivity
        Hbc_matrix += partial_Hbc_matrix

    return Hbc_matrix

def generate_Hbc_matrix(
    element: Element,
    globalData,
) -> np.matrix:
    Hbc_matrix = np.zeros((4, 4))
    shape_functions_at_boundary = generate_shape_functions_at_boundary()

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0)
    ]

    for edge_index, node_ids in enumerate(edges):
        Hbc_matrix += generate_Hbc_matrix_for_side(
            element,
            node_ids,
            globalData,
            shape_functions_at_boundary[edge_index, :, :]
        )

    return Hbc_matrix