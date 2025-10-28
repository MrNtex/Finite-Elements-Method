import numpy as np
from typing import List

from jacobian import Jacobian
from config import NUMBER_OF_INTEGRATION_POINTS
from gauss_integration import GAUSS_QUADRATURE
from fem_types import GlobalData

def transform_local_derivatives_to_global(
    dN_d_epsilon: np.matrix,
    dN_d_eta: np.matrix,
    jacobians: List[Jacobian],
) -> tuple[np.matrix, np.matrix]:
    dN_d_x = np.zeros(dN_d_epsilon.shape)
    dN_d_y = np.zeros(dN_d_eta.shape)

    for integration_point_index in range(NUMBER_OF_INTEGRATION_POINTS**2):
        for i in range(dN_d_epsilon.shape[1]): # Number of rows columns is number of shape functions (4)
            dN_d_x[integration_point_index, i] = (jacobians[integration_point_index].invJ[0, 0] * dN_d_epsilon[integration_point_index, i] +
                                                    jacobians[integration_point_index].invJ[0, 1] * dN_d_eta[integration_point_index, i])
            dN_d_y[integration_point_index, i] = (jacobians[integration_point_index].invJ[1, 0] * dN_d_epsilon[integration_point_index, i] +
                                                    jacobians[integration_point_index].invJ[1, 1] * dN_d_eta[integration_point_index, i])

    return dN_d_x, dN_d_y

def generate_H_matrix(
    dN_d_x: np.matrix,
    dN_d_y: np.matrix,
    jacobians: List[Jacobian],
    globalData: GlobalData,
) -> np.matrix:
    H_matrix = np.zeros((4, 4))

    for ip_index_x in range(NUMBER_OF_INTEGRATION_POINTS):
        for ip_index_y in range(NUMBER_OF_INTEGRATION_POINTS):
            ip_index = ip_index_y * NUMBER_OF_INTEGRATION_POINTS + ip_index_x
            weight = GAUSS_QUADRATURE[NUMBER_OF_INTEGRATION_POINTS]["weights"][ip_index_x] * GAUSS_QUADRATURE[NUMBER_OF_INTEGRATION_POINTS]["weights"][ip_index_y]
            detJ = jacobians[ip_index].detJ

            partial_H_matrix = (
                np.outer(dN_d_x[ip_index, :], dN_d_x[ip_index, :]) +
                np.outer(dN_d_y[ip_index, :], dN_d_y[ip_index, :])
            ) * weight * detJ * globalData.Conductivity

            H_matrix += partial_H_matrix

    return H_matrix