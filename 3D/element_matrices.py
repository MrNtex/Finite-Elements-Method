import numpy as np
from typing import List

from jacobian import Jacobian, UniversalJacobian
from config import NUMBER_OF_INTEGRATION_POINTS
from gauss_integration import GAUSS_QUADRATURE
from fem_types import Element, GlobalData

def transform_local_derivatives_to_global(
    dN_d_xi: np.matrix,
    dN_d_eta: np.matrix,
    dN_d_zeta: np.matrix,
    jacobians: List[Jacobian],
) -> tuple[np.matrix, np.matrix, np.matrix]:
    dN_d_x = np.zeros(dN_d_xi.shape)
    dN_d_y = np.zeros(dN_d_eta.shape)
    dN_d_z = np.zeros(dN_d_zeta.shape)

    num_points = dN_d_xi.shape[0] 
    for i in range(num_points):
        invJ = jacobians[i].invJ
        for n in range(8):
            dN_d_x[i, n] = (invJ[0, 0] * dN_d_xi[i, n] +
                            invJ[0, 1] * dN_d_eta[i, n] +
                            invJ[0, 2] * dN_d_zeta[i, n])
            dN_d_y[i, n] = (invJ[1, 0] * dN_d_xi[i, n] +
                            invJ[1, 1] * dN_d_eta[i, n] +
                            invJ[1, 2] * dN_d_zeta[i, n])
            dN_d_z[i, n] = (invJ[2, 0] * dN_d_xi[i, n] +
                            invJ[2, 1] * dN_d_eta[i, n] +
                            invJ[2, 2] * dN_d_zeta[i, n])

    return dN_d_x, dN_d_y, dN_d_z

def calculate_element_matrices(
    dN_d_x: np.matrix,
    dN_d_y: np.matrix,
    dN_d_z: np.matrix,
    N_functions: np.matrix,
    jacobians: List[Jacobian],
    globalData: GlobalData,
    element: Element
) -> tuple[np.matrix, np.matrix, np.matrix]:
    H_matrix = np.zeros((8, 8))
    C_matrix = np.zeros((8, 8))
    P_source_vector = np.zeros(8)

    density = globalData.Density
    specific_heat = globalData.SpecificHeat
    
    weights = GAUSS_QUADRATURE[NUMBER_OF_INTEGRATION_POINTS]["weights"]

    for i in range(NUMBER_OF_INTEGRATION_POINTS):       # zeta
        for j in range(NUMBER_OF_INTEGRATION_POINTS):   # eta
            for k in range(NUMBER_OF_INTEGRATION_POINTS): # xi
                ip_index = i * (NUMBER_OF_INTEGRATION_POINTS**2) + j * NUMBER_OF_INTEGRATION_POINTS + k
                weight = weights[k] * weights[j] * weights[i]
                
                detJ = jacobians[ip_index].detJ
                partial_H = (
                    np.outer(dN_d_x[ip_index, :], dN_d_x[ip_index, :]) +
                    np.outer(dN_d_y[ip_index, :], dN_d_y[ip_index, :]) +
                    np.outer(dN_d_z[ip_index, :], dN_d_z[ip_index, :])
                ) * element.k * weight * detJ

                partial_C = (
                    np.outer(N_functions[ip_index, :], N_functions[ip_index, :])
                ) * element.rho * element.cp * weight * detJ

                H_matrix += partial_H
                C_matrix += partial_C
                P_source_vector += N_functions[ip_index, :] * element.Q * weight * detJ

    return H_matrix, C_matrix, P_source_vector