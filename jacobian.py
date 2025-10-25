from dataclasses import dataclass
from typing import List
import numpy as np

from config import NUMBER_OF_INTEGRATION_POINTS
from gauss_integration import GAUSS_QUADRATURE
from fem_types import Jacobian, Node, Element, Grid

@dataclass
class UniversalJacobian:
  dN_d_epsilon: np.matrix
  dN_d_eta: np.matrix

  def __init__(self):
    self.dN_d_epsilon = np.zeros((NUMBER_OF_INTEGRATION_POINTS**2, 4))
    self.dN_d_eta = np.zeros((NUMBER_OF_INTEGRATION_POINTS**2, 4))

    nodes = np.array(GAUSS_QUADRATURE[NUMBER_OF_INTEGRATION_POINTS]["nodes"])

    for j, eta in enumerate(nodes):
      for i, xi in enumerate(nodes):
          idx = j * NUMBER_OF_INTEGRATION_POINTS + i

          print(i, j, xi, eta, idx)
          self.dN_d_epsilon[idx, 0] = -0.25 * (1 - eta)
          self.dN_d_epsilon[idx, 1] =  0.25 * (1 - eta)
          self.dN_d_epsilon[idx, 2] =  0.25 * (1 + eta)
          self.dN_d_epsilon[idx, 3] = -0.25 * (1 + eta)

          self.dN_d_eta[idx, 0] = -0.25 * (1 - xi)
          self.dN_d_eta[idx, 1] = -0.25 * (1 + xi)
          self.dN_d_eta[idx, 2] =  0.25 * (1 + xi)
          self.dN_d_eta[idx, 3] =  0.25 * (1 - xi)

def calculate_jacobian_for_finite_element(
    element: Element,
    grid: Grid,
    universal_jacobian: UniversalJacobian,
) -> List[Jacobian]:
    nodes_x = np.array([grid.nodes[n_id - 1].x for n_id in element.node_ids])
    nodes_y = np.array([grid.nodes[n_id - 1].y for n_id in element.node_ids])
    jacobians = []

    for integration_point_index in range(NUMBER_OF_INTEGRATION_POINTS**2):
      dN_d_epsilon = universal_jacobian.dN_d_epsilon[integration_point_index, :]
      dN_d_eta = universal_jacobian.dN_d_eta[integration_point_index, :]

      J = np.zeros((2, 2))
      J[0, 0] = np.sum(dN_d_epsilon * nodes_x)
      J[0, 1] = np.sum(dN_d_epsilon * nodes_y)
      J[1, 0] = np.sum(dN_d_eta * nodes_x)
      J[1, 1] = np.sum(dN_d_eta * nodes_y)

      jacobians.append(Jacobian(J=J, invJ=np.linalg.inv(J), detJ=np.linalg.det(J)))

    return jacobians