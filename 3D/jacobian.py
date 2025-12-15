from dataclasses import dataclass
from typing import List
import numpy as np

from config import NUMBER_OF_INTEGRATION_POINTS
from gauss_integration import GAUSS_QUADRATURE
from fem_types import Jacobian, Node, Element, Grid

@dataclass
class UniversalJacobian:
  num_points: int = NUMBER_OF_INTEGRATION_POINTS ** 3
  dN_d_epsilon: np.matrix
  dN_d_eta: np.matrix
  dN_d_zeta: np.matrix
  N_functions: np.matrix

  def __init__(self):
    self.dN_d_epsilon = np.zeros((self.num_points, 8))
    self.dN_d_eta = np.zeros((self.num_points, 8))
    self.dN_d_zeta = np.zeros((self.num_points, 8))
    self.N_functions = np.zeros((self.num_points, 8))

    integration_nodes = np.array(GAUSS_QUADRATURE[NUMBER_OF_INTEGRATION_POINTS]["nodes"])

    idx = 0
    for zeta in integration_nodes:      # Z Axis (Height)
      for eta in integration_nodes:   # Y Axis (Depth)
        for xi in integration_nodes:# X Axis (Width)
                
          # --- LOWER NODES (zeta = -1) ---
          
          # Node 0 (-1, -1, -1)
          self.dN_d_xi[idx, 0]   = -0.125 * (1 - eta) * (1 - zeta)
          self.dN_d_eta[idx, 0]  = -0.125 * (1 - xi)  * (1 - zeta)
          self.dN_d_zeta[idx, 0] = -0.125 * (1 - xi)  * (1 - eta)
          self.N_functions[idx, 0] = 0.125 * (1 - xi) * (1 - eta) * (1 - zeta)

          # Node 1 (+1, -1, -1)
          self.dN_d_xi[idx, 1]   =  0.125 * (1 - eta) * (1 - zeta)
          self.dN_d_eta[idx, 1]  = -0.125 * (1 + xi)  * (1 - zeta)
          self.dN_d_zeta[idx, 1] = -0.125 * (1 + xi)  * (1 - eta)
          self.N_functions[idx, 1] = 0.125 * (1 + xi) * (1 - eta) * (1 - zeta)

          # Node 2 (+1, +1, -1)
          self.dN_d_xi[idx, 2]   =  0.125 * (1 + eta) * (1 - zeta)
          self.dN_d_eta[idx, 2]  =  0.125 * (1 + xi)  * (1 - zeta)
          self.dN_d_zeta[idx, 2] = -0.125 * (1 + xi)  * (1 + eta)
          self.N_functions[idx, 2] = 0.125 * (1 + xi) * (1 + eta) * (1 - zeta)

          # Node 3 (-1, +1, -1)
          self.dN_d_xi[idx, 3]   = -0.125 * (1 + eta) * (1 - zeta)
          self.dN_d_eta[idx, 3]  =  0.125 * (1 - xi)  * (1 - zeta)
          self.dN_d_zeta[idx, 3] = -0.125 * (1 - xi)  * (1 + eta)
          self.N_functions[idx, 3] = 0.125 * (1 - xi) * (1 + eta) * (1 - zeta)

          # --- UPPER NODES (zeta = +1) ---

          # Node 4 (-1, -1, +1)
          self.dN_d_xi[idx, 4]   = -0.125 * (1 - eta) * (1 + zeta)
          self.dN_d_eta[idx, 4]  = -0.125 * (1 - xi)  * (1 + zeta)
          self.dN_d_zeta[idx, 4] =  0.125 * (1 - xi)  * (1 - eta)
          self.N_functions[idx, 4] = 0.125 * (1 - xi) * (1 - eta) * (1 + zeta)

          # Node 5 (+1, -1, +1)
          self.dN_d_xi[idx, 5]   =  0.125 * (1 - eta) * (1 + zeta)
          self.dN_d_eta[idx, 5]  = -0.125 * (1 + xi)  * (1 + zeta)
          self.dN_d_zeta[idx, 5] =  0.125 * (1 + xi)  * (1 - eta)
          self.N_functions[idx, 5] = 0.125 * (1 + xi) * (1 - eta) * (1 + zeta)

          # Node 6 (+1, +1, +1)
          self.dN_d_xi[idx, 6]   =  0.125 * (1 + eta) * (1 + zeta)
          self.dN_d_eta[idx, 6]  =  0.125 * (1 + xi)  * (1 + zeta)
          self.dN_d_zeta[idx, 6] =  0.125 * (1 + xi)  * (1 + eta)
          self.N_functions[idx, 6] = 0.125 * (1 + xi) * (1 + eta) * (1 + zeta)

          # Node 7 (-1, +1, +1)
          self.dN_d_xi[idx, 7]   = -0.125 * (1 + eta) * (1 + zeta)
          self.dN_d_eta[idx, 7]  =  0.125 * (1 - xi)  * (1 + zeta)
          self.dN_d_zeta[idx, 7] =  0.125 * (1 - xi)  * (1 + eta)
          self.N_functions[idx, 7] = 0.125 * (1 - xi) * (1 + eta) * (1 + zeta)

          idx += 1

def calculate_jacobian_for_finite_element(
    element: Element,
    grid: Grid,
    universal_jacobian: UniversalJacobian,
) -> List[Jacobian]:
    nodes_x = np.array([grid.nodes[n_id - 1].x for n_id in element.node_ids])
    nodes_y = np.array([grid.nodes[n_id - 1].y for n_id in element.node_ids])
    nodes_z = np.array([grid.nodes[n_id - 1].z for n_id in element.node_ids])
    jacobians = []

    for integration_point_index in range(universal_jacobian.num_points):
      dN_d_xi = universal_jacobian.dN_d_epsilon[integration_point_index, :]
      dN_d_eta = universal_jacobian.dN_d_eta[integration_point_index, :]
      dN_d_zeta = universal_jacobian.dN_d_zeta[integration_point_index, :]

      J = np.zeros((3, 3))
      J[0, 0] = np.sum(dN_d_xi * nodes_x) # dx/d_xi
      J[0, 1] = np.sum(dN_d_xi * nodes_y) # dy/d_xi
      J[0, 2] = np.sum(dN_d_xi * nodes_z) # dz/d_xi
      
      J[1, 0] = np.sum(dN_d_eta * nodes_x) # dx/d_eta
      J[1, 1] = np.sum(dN_d_eta * nodes_y) # dy/d_eta
      J[1, 2] = np.sum(dN_d_eta * nodes_z) # dz/d_eta

      J[2, 0] = np.sum(dN_d_zeta * nodes_x) # dx/d_zeta
      J[2, 1] = np.sum(dN_d_zeta * nodes_y) # dy/d_zeta
      J[2, 2] = np.sum(dN_d_zeta * nodes_z) # dz/d_zeta
      jacobians.append(Jacobian(J=J, invJ=np.linalg.inv(J), detJ=np.linalg.det(J)))

    return jacobians
